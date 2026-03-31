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


def find_local_maxima_3d(noise_map, threshold=0.5, min_distance=5):
    """
    Находит локальные максимумы в 3D карте шума.
    """
    from scipy.ndimage import maximum_filter
    
    # Нормализуем шум в [0, 1]
    noise_norm = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
    
    # Фильтр локальных максимумов
    neighborhood_size = min_distance * 2 + 1
    local_max = maximum_filter(noise_norm, size=neighborhood_size) == noise_norm
    
    # Применяем порог
    maxima_mask = local_max & (noise_norm > threshold)
    
    # Получаем координаты максимумов
    maxima_coords = np.argwhere(maxima_mask)
    
    return maxima_coords, noise_norm


def calculate_sphere_radii(maxima_coords, noise_map, shape, max_radius=8, overlap_factor=0.15):
    """
    Рассчитывает радиусы сфер так, чтобы они касались соседних с небольшим перекрытием.
    """
    if len(maxima_coords) == 0:
        return []
    
    radii = []
    
    for i, center in enumerate(maxima_coords):
        # Находим расстояния до всех других центров
        distances = []
        for j, other_center in enumerate(maxima_coords):
            if i != j:
                dist = np.sqrt(np.sum((center - other_center)**2))
                distances.append(dist)
        
        if len(distances) > 0:
            # Минимальное расстояние до соседа
            min_dist = min(distances)
            # Радиус = половина минимального расстояния * (1 + overlap_factor)
            radius = min_dist / 2.0 * (1 + overlap_factor)
            radius = min(radius, max_radius)  # Ограничиваем максимальный радиус
            radius = max(radius, 2)  # Минимальный радиус
        else:
            radius = max_radius
        
        radii.append(radius)
    
    return radii


def create_cube_with_spheres(shape, voxel_size=0.5, wall_thickness_mm=1.0, 
                             perlin_scale=30.0, max_radius_voxels=8, overlap_factor=0.15):
    """
    Создаёт куб с внутренними сферами на основе шума Перлина.
    Учитывает требования 3D печати (опоры для висящих частей).
    """
    wall_thickness_vox = int(wall_thickness_mm / voxel_size)
    
    # Генерируем шум Перлина
    print("📊 Генерация шума Перлина...")
    noise_map = generate_perlin_noise_3d(shape, scale=perlin_scale)
    
    # Находим локальные максимумы
    print("🔍 Поиск локальных максимумов...")
    maxima_coords, noise_norm = find_local_maxima_3d(noise_map, threshold=0.4, min_distance=5)
    
    print(f"   Найдено {len(maxima_coords)} локальных максимумов")
    
    # Рассчитываем радиусы сфер
    print("📏 Расчёт радиусов сфер...")
    radii = calculate_sphere_radii(maxima_coords, noise_map, shape, 
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
    inner_region = np.zeros(shape, dtype=bool)
    inner_region[wall_thickness_vox:-wall_thickness_vox,
                 wall_thickness_vox:-wall_thickness_vox,
                 wall_thickness_vox:-wall_thickness_vox] = True
    
    # Вырезаем сферы
    print("🔮 Генерация сфер...")
    sphere_mask = np.zeros(shape, dtype=bool)
    
    for i, (center, radius) in enumerate(zip(maxima_coords, radii)):
        x, y, z = center
        xx, yy, zz = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist = np.sqrt((xx - x)**2 + (yy - y)**2 + (zz - z)**2)
        sphere = dist <= radius
        
        # Обрезаем сферу по внешней стенке
        sphere = sphere & inner_region
        
        sphere_mask = sphere_mask | sphere
    
    # Вырезаем сферы из материала
    mask = mask & ~sphere_mask
    
    # Добавляем опоры для 3D печати (Z-axis printing)
    print("🏗️  Генерация опор для 3D печати...")
    mask = add_support_structures(mask, voxel_size)
    
    return mask, maxima_coords, radii


def add_support_structures(mask, voxel_size=0.5, min_overhang_angle=45, support_thickness_vox=2):
    """
    Добавляет опорные структуры для висящих частей (учёт 3D печати).
    """
    shape = mask.shape
    result = mask.copy()
    
    # Для каждого слоя сверху вниз
    for z in range(shape[2] - 2, 0, -1):
        # Текущий слой
        current_layer = result[:, :, z]
        # Слой ниже
        below_layer = result[:, :, z - 1]
        
        # Находим висящие части (есть материал сейчас, но нет ниже)
        overhang = current_layer & ~below_layer
        
        # Если есть висящие части, добавляем опоры
        if np.any(overhang):
            # Расширяем опоры для лучшей поддержки
            overhang_dilated = ndimage.binary_dilation(overhang, iterations=support_thickness_vox)
            # Добавляем опоры только там, где нет материала
            support_region = overhang_dilated & ~below_layer
            result[support_region, z - 1] = True
    
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
    resolution = (80, 80, 80)  # Размер куба в вокселях
    voxel_size = 0.5  # мм
    wall_thickness_mm = 1.0  # мм (минимум для 3D печати)
    
    # Параметры шума Перлина
    perlin_scale = 30.0  # Частота шума (меньше = чаще максимумы)
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0
    
    # Параметры сфер
    max_radius_voxels = 8
    overlap_factor = 0.15  # 15% перекрытие сфер
    
    print(f"📐 Параметры:")
    print(f"   Разрешение: {resolution} вокселей")
    print(f"   Размер вокселя: {voxel_size} мм")
    print(f"   Толщина стенок: {wall_thickness_mm} мм")
    print(f"   Scale шума: {perlin_scale}")
    print(f"   Макс. радиус сфер: {max_radius_voxels * voxel_size} мм")
    print(f"   Перекрытие сфер: {overlap_factor * 100}%\n")
    
    # Генерация
    mask, maxima_coords, radii = create_cube_with_spheres(
        shape=resolution,
        voxel_size=voxel_size,
        wall_thickness_mm=wall_thickness_mm,
        perlin_scale=perlin_scale,
        max_radius_voxels=max_radius_voxels,
        overlap_factor=overlap_factor
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
