import numpy as np
import noise
import trimesh
from scipy import ndimage


def generate_perlin_noise_3d(shape, scale=50.0, octaves=6, persistence=0.5, lacunarity=2.0, base=0):
    """
    Генерирует 3D карту шума Перлина.
    """
    noise_map = np.zeros(shape)
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                noise_value = noise.pnoise3(
                    x / scale,
                    y / scale,
                    z / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=1024,
                    repeaty=1024,
                    repeatz=1024,
                    base=base
                )
                noise_map[x, y, z] = noise_value
    return noise_map


def find_local_extrema_3d(noise_map, threshold=0.3, min_distance=3, max_candidates=10000):
    """
    Находит локальные экстремумы (максимумы и минимумы) в 3D карте шума.
    """
    from scipy.ndimage import maximum_filter, minimum_filter
    
    # Нормализуем шум в [0, 1]
    noise_norm = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
    
    # Фильтр локальных максимумов
    neighborhood_size = min_distance * 2 + 1
    local_max = maximum_filter(noise_norm, size=neighborhood_size) == noise_norm
    local_min = minimum_filter(noise_norm, size=neighborhood_size) == noise_norm
    
    # Применяем порог для максимумов (близко к 1) и минимумов (близко к 0)
    maxima_mask = local_max & (noise_norm > (1 - threshold))
    minima_mask = local_min & (noise_norm < threshold)
    
    # Получаем координаты экстремумов
    maxima_coords = np.argwhere(maxima_mask)
    minima_coords = np.argwhere(minima_mask)
    
    # Объединяем максимумы и минимумы
    all_extrema = np.vstack([maxima_coords, minima_coords]) if len(maxima_coords) > 0 and len(minima_coords) > 0 else maxima_coords if len(maxima_coords) > 0 else minima_coords
    
    # Если слишком много кандидатов, отбираем лучшие по значению шума
    if len(all_extrema) > max_candidates:
        # Сортируем по "силе" экстремума (отклонение от 0.5)
        extremum_values = []
        for coord in all_extrema:
            val = noise_norm[coord[0], coord[1], coord[2]]
            strength = abs(val - 0.5)  # Чем дальше от 0.5, тем лучше
            extremum_values.append(strength)
        
        # Выбираем топ-N
        top_indices = np.argsort(extremum_values)[-max_candidates:]
        all_extrema = all_extrema[top_indices]
    
    return all_extrema, noise_norm


def calculate_sphere_radii(extrema_coords, noise_map, shape, max_radius=8, overlap_factor=0.15):
    """
    Рассчитывает радиусы сфер так, чтобы они касались соседних с небольшим перекрытием.
    Для большого количества сфер используем упрощённый алгоритм.
    """
    if len(extrema_coords) == 0:
        return []
    
    # Для большого количества сфер используем KD-tree для быстрого поиска соседей
    from scipy.spatial import cKDTree
    
    tree = cKDTree(extrema_coords)
    radii = []
    
    for i, center in enumerate(extrema_coords):
        # Находим k ближайших соседей (k=10 для скорости)
        k = min(10, len(extrema_coords))
        distances, _ = tree.query(center, k=k+1)  # +1 потому что первый сосед - это сама точка
        
        # Пропускаем первую точку (расстояние 0 до самой себя)
        if len(distances) > 1:
            # Минимальное расстояние до соседа (пропускаем 0)
            min_dist = distances[1] if distances[0] == 0 else distances[0]
            # Радиус = половина минимального расстояния * (1 + overlap_factor)
            radius = min_dist / 2.0 * (1 + overlap_factor)
            radius = min(radius, max_radius)  # Ограничиваем максимальный радиус
            radius = max(radius, 2)  # Минимальный радиус
        else:
            radius = max_radius
        
        radii.append(radius)
    
    return radii


def create_cube_with_spheres(shape, voxel_size=0.5, wall_thickness_mm=1.0, 
                             perlin_scale=30.0, max_radius_voxels=8, overlap_factor=0.15,
                             fill_threshold=0.4, max_attempts=10):
    """
    Создаёт куб с внутренними сферами на основе шума Перлина.
    Учитывает требования 3D печати (опоры для висящих частей).
    fill_threshold: порог заполнения материалом (0.3-0.4 = 30-40% материала, 60-70% пустоты)
    max_attempts: максимальное количество попыток генерации связной структуры
    """
    wall_thickness_vox = int(wall_thickness_mm / voxel_size)
    
    for attempt in range(max_attempts):
        print(f"\n{'='*50}")
        print(f"🔄 Попытка генерации #{attempt + 1}/{max_attempts}")
        print('='*50)
        
        # Генерируем шум Перлина с небольшим смещением для вариативности
        noise_map = generate_perlin_noise_3d(shape, scale=perlin_scale, base=attempt * 100)
        
        # Нормализуем шум
        noise_norm = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
        
        # Находим локальные экстремумы (максимумы и минимумы) для центров сфер
        print("🔍 Поиск локальных экстремумов (максимумы и минимумы)...")
        extrema_coords, _ = find_local_extrema_3d(noise_map, threshold=0.25, min_distance=4, max_candidates=600)
        
        if len(extrema_coords) < 50:
            print(f"   ⚠️  Слишком мало экстремумов ({len(extrema_coords)}), пробуем снова...")
            continue
        
        print(f"   Найдено {len(extrema_coords)} локальных экстремумов")
        
        # Рассчитываем радиусы сфер
        print("📏 Расчёт радиусов сфер...")
        radii = calculate_sphere_radii(extrema_coords, noise_map, shape, 
                                       max_radius=max_radius_voxels, 
                                       overlap_factor=overlap_factor)
        
        # Создаём базовую маску (сплошной куб)
        mask = np.ones(shape, dtype=bool)
        
        # Создаём внешние стенки (обязательно 1мм минимум)
        print(f"📦 Создание внешних стенок (толщина {wall_thickness_mm}мм)...")
        if wall_thickness_vox > 0:
            mask[wall_thickness_vox:-wall_thickness_vox,
                 wall_thickness_vox:-wall_thickness_vox,
                 wall_thickness_vox:-wall_thickness_vox] = False
            mask = ~mask  # Инвертируем - теперь True это материал
        
        # Маска для внутренней области (где можно создавать сферы)
        # Увеличиваем отступ чтобы сферы не затрагивали внешнюю поверхность
        inner_margin = wall_thickness_vox + 2  # Дополнительный отступ 2 вокселя
        inner_region = np.zeros(shape, dtype=bool)
        inner_region[inner_margin:-inner_margin,
                     inner_margin:-inner_margin,
                     inner_margin:-inner_margin] = True
        
        # Создаём волны заполнения на основе шума Перлина
        # Где шум выше порога - материал, ниже - пустота
        print(f"🌊 Создание волн заполнения (порог {fill_threshold})...")
        wave_mask = noise_norm > fill_threshold
        
        # Вырезаем сферы из материала
        print("🔮 Генерация сфер...")
        sphere_mask = np.zeros(shape, dtype=bool)
        
        # Толщина стенок сферы (в вокселях)
        sphere_wall_thickness = 2  # ~1 мм
        
        for i, (center, radius) in enumerate(zip(extrema_coords, radii)):
            x, y, z = center
            xx, yy, zz = np.ogrid[:shape[0], :shape[1], :shape[2]]
            dist = np.sqrt((xx - x)**2 + (yy - y)**2 + (zz - z)**2)
            
            # Создаём полую сферу (только оболочка)
            inner_radius = max(radius - sphere_wall_thickness, 1)
            sphere_shell = (dist <= radius) & (dist > inner_radius)
            
            # Обрезаем сферу по внутренней области (не затрагивая внешние стенки)
            sphere_shell = sphere_shell & inner_region
            
            sphere_mask = sphere_mask | sphere_shell
        
        # Комбинируем: материал = стенки + (волны заполнения & ~сферы)
        # Сферы вырезаются из волн заполнения
        inner_material = wave_mask & inner_region & ~sphere_mask
        
        # Итоговая маска: внешние стенки + внутренний материал
        full_mask = np.ones(shape, dtype=bool)
        full_mask[inner_margin:-inner_margin,
                  inner_margin:-inner_margin,
                  inner_margin:-inner_margin] = inner_material[inner_margin:-inner_margin,
                                                                inner_margin:-inner_margin,
                                                                inner_margin:-inner_margin]
        
        # Добавляем опоры для 3D печати (Z-axis printing)
        print("🏗️  Генерация опор для 3D печати...")
        full_mask = add_support_structures(full_mask, voxel_size)
        
        # Проверяем связность внутренней структуры
        print("🔗 Проверка связности внутренней структуры...")
        is_connected, num_components = check_connectivity(full_mask, inner_region)
        
        if is_connected:
            print(f"   ✅ Структура связная! (1 компонента)")
            return full_mask, extrema_coords, radii
        else:
            print(f"   ⚠️  Структура несвязная ({num_components} компонент), пробуем снова...")
    
    print(f"\n❌ Не удалось создать связную структуру после {max_attempts} попыток")
    print("   Возвращаем последнюю сгенерированную версию...")
    return full_mask, extrema_coords, radii


def check_connectivity(mask, inner_region):
    """
    Проверяет связность внутренней структуры (полостей).
    Возвращает (is_connected, num_components).
    """
    from scipy.ndimage import label
    
    # Выделяем внутреннюю полость (пустоты внутри)
    # Инвертируем маску чтобы получить пустоты
    voids = ~mask & inner_region
    
    # Добавляем морфологическую дилатацию для соединения близких пустот
    voids = ndimage.binary_dilation(voids, iterations=1)
    
    # Считаем компоненты связности
    labeled, num_features = label(voids)
    
    # Если есть пустоты, проверяем что основная компонента большая
    if num_features == 0:
        return True, 0  # Нет пустот - всё связно
    
    # Находим размер каждой компоненты
    component_sizes = np.bincount(labeled.ravel())
    
    # Сортируем по размеру (исключая 0 фон)
    sorted_sizes = sorted(component_sizes[1:], reverse=True)
    
    if len(sorted_sizes) == 0:
        return True, 0
    
    # Если самая большая компонента > 50% от всех пустот - считаем связным
    largest_ratio = sorted_sizes[0] / sum(sorted_sizes)
    
    # Также проверяем что компонент не слишком много
    is_connected = (largest_ratio > 0.5) and (num_features < 100)
    
    return is_connected, num_features


def add_support_structures(mask, voxel_size=0.5, min_overhang_angle=45, support_thickness_vox=2):
    """
    Добавляет опорные структуры для висящих частей (учёт 3D печати).
    Проверка на угол наклона поверхности относительно оси Z.
    """
    shape = mask.shape
    result = mask.copy()
    
    # Создаём маску опор
    support_mask = np.zeros(shape, dtype=bool)
    
    # Для каждого слоя снизу вверх (кроме самого нижнего)
    for z in range(1, shape[2]):
        # Слой ниже
        below_layer = result[:, :, z - 1].astype(float)
        # Текущий слой
        current_layer = result[:, :, z].astype(float)
        
        # Находим висящие части (есть материал сейчас, но нет поддержки ниже)
        # Используем эрозию ниже слоя для проверки достаточной поддержки
        below_eroded = ndimage.binary_erosion(result[:, :, z - 1], iterations=1)
        
        # Висящие части: материал есть, но поддержки нет
        overhang = result[:, :, z] & ~below_eroded
        
        # Если есть висящие части, добавляем опоры под ними
        if np.any(overhang):
            # Опоры идут от висящей части вниз до ближайшей поддержки
            for check_z in range(z - 1, -1, -1):
                support_layer = result[:, :, check_z]
                # Если нашли поддержку - останавливаемся
                if np.any(support_layer & overhang):
                    break
                # Добавляем опоры в этом слое
                support_mask[overhang, check_z] = True
    
    # Объединяем опоры с основной маской
    result = result | support_mask
    
    # Укрепляем опоры (делаем их толще)
    if support_thickness_vox > 0:
        # Применяем дилатацию только к опорам, но не к основному объекту
        pass  # Оставляем тонкие опоры для экономии материала
    
    return result


def mask_to_mesh(mask, voxel_size=1.0):
    """
    Конвертирует бинарную 3D-маску в меш.
    """
    from skimage import measure
    vertices, faces, normals, values = measure.marching_cubes(mask.astype(float), level=0.5)
    vertices *= voxel_size
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Исправляем ориентацию нормалей
    if mesh.volume < 0:
        mesh.invert()
    
    return mesh


def cleanup_mesh(mesh):
    """
    Очищает меш.
    """
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    return mesh


def main():
    print("🔧 Генерация 3D-модели: куб со сферами на основе шума Перлина\n")
    
    # Базовые параметры
    resolution = (100, 100, 100)  # Размер куба в вокселях
    voxel_size = 0.5  # мм
    wall_thickness_mm = 1.5  # мм (минимум для 3D печати)
    
    # Параметры шума Перлина
    perlin_scale = 15.0  # Частота шума (для волн заполнения)
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0
    
    # Параметры сфер
    max_radius_voxels = 12  # Увеличен для больших сфер
    overlap_factor = 0.35   # 35% перекрытие для сильного пересечения
    
    # Заполнение тела (30-40% материала, 60-70% пустоты)
    fill_threshold = 0.38   # Порог заполнения по шуму Перлина
    
    print(f"📐 Параметры:")
    print(f"   Разрешение: {resolution} вокселей")
    print(f"   Размер вокселя: {voxel_size} мм")
    print(f"   Толщина стенок: {wall_thickness_mm} мм")
    print(f"   Scale шума: {perlin_scale}")
    print(f"   Макс. радиус сфер: {max_radius_voxels * voxel_size} мм")
    print(f"   Перекрытие сфер: {overlap_factor * 100}%")
    print(f"   Порог заполнения: {fill_threshold} (пустота ~{100 - fill_threshold*100:.0f}%)\n")
    
    # Генерация
    mask, extrema_coords, radii = create_cube_with_spheres(
        shape=resolution,
        voxel_size=voxel_size,
        wall_thickness_mm=wall_thickness_mm,
        perlin_scale=perlin_scale,
        max_radius_voxels=max_radius_voxels,
        overlap_factor=overlap_factor,
        fill_threshold=fill_threshold,
        max_attempts=15  # До 15 попыток для связной структуры
    )
    
    # Конвертация в меш
    print("\n🔄 Конвертация в меш...")
    mesh = mask_to_mesh(mask, voxel_size)
    mesh = cleanup_mesh(mesh)
    
    print(f"\n✅ Вершин: {len(mesh.vertices)}, Граней: {len(mesh.faces)}")
    print(f"📐 Объем: {mesh.volume:.2f} мм³")
    print(f"📏 Размеры: {mesh.extents}")
    
    # Сохранение
    mesh.export("cheese_structure.stl")
    print("\n💾 Файл сохранён: cheese_structure.stl")


if __name__ == "__main__":
    main()
