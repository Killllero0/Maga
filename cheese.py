import numpy as np
import noise
import trimesh
from scipy import ndimage

def create_perlin_noise_shape(shape, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, threshold=0.3, wall_thickness=4):
    """
    Создаёт 3D-маску с внешними стенками и внутренними пещерами на основе шума Перлина.
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
                    base=0
                )
                noise_map[x, y, z] = noise_value

    # Нормализуем шум в диапазон [-1, 1] -> [0, 1]
    noise_normalized = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
    
    # Создаём маску пустот на основе шума Перлина (пещеры)
    void_mask = noise_normalized > threshold
    
    # Создаём внешнюю оболочку (стенки)
    # Внутренняя область где можно создавать пустоты
    inner_region = np.zeros(shape, dtype=bool)
    inner_region[wall_thickness:-wall_thickness, 
                 wall_thickness:-wall_thickness, 
                 wall_thickness:-wall_thickness] = True
    
    # Расширяем пустоты чтобы они соединялись в единую систему пещер
    void_mask = ndimage.binary_dilation(void_mask, iterations=3)
    
    # Оставляем пустоты только во внутренней области (не затрагивая стенки)
    void_mask = void_mask & inner_region
    
    # Создаём основное тело - сплошное
    mask = np.ones(shape, dtype=bool)
    
    # Вырезаем пещеры из тела
    mask = mask & ~void_mask
    
    return mask

def create_spherical_voids(mask, num_voids=50, min_radius=3, max_radius=8, wall_thickness=4):
    """
    Добавляет дополнительные сферические пустоты внутри тела, не затрагивая внешние стенки.
    """
    shape = mask.shape
    result = mask.copy()
    np.random.seed(42)

    # Внутренняя область (где можно создавать пустоты)
    inner_mask = np.zeros(shape, dtype=bool)
    inner_mask[wall_thickness:-wall_thickness, 
               wall_thickness:-wall_thickness, 
               wall_thickness:-wall_thickness] = True

    for _ in range(num_voids):
        radius = np.random.randint(min_radius, max_radius + 1)
        # Центрируем пустоты внутри тела с отступом от краёв
        margin = wall_thickness + radius
        x = np.random.randint(margin, shape[0] - margin)
        y = np.random.randint(margin, shape[1] - margin)
        z = np.random.randint(margin, shape[2] - margin)

        xx, yy, zz = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist = np.sqrt((xx - x)**2 + (yy - y)**2 + (zz - z)**2)
        sphere = dist <= radius
        
        # Вырезаем сферу только если она внутри внутренней области
        result[sphere & inner_mask] = False

    return result

def mask_to_mesh(mask, voxel_size=1.0):
    """
    Конвертирует бинарную 3D-маску в меш.
    """
    from skimage import measure
    vertices, faces, normals, values = measure.marching_cubes(mask.astype(float), level=0.5)
    vertices *= voxel_size
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Исправляем ориентацию нормалей (объём должен быть положительным)
    if mesh.volume < 0:
        mesh.invert()
    
    return mesh

def cleanup_mesh(mesh):
    """
    Очищает меш совместимым способом.
    """
    # Объединяем близкие вершины
    mesh.merge_vertices()

    # Удаляем вырожденные грани
    mesh.update_faces(mesh.nondegenerate_faces())

    # Удаляем бесконечные значения
    mesh.remove_infinite_values()

    # Удаляем вершины без граней
    mesh.remove_unreferenced_vertices()

    return mesh

def main():
    print("🔧 Генерация 3D-модели с шумом Перлина и внутренними пещерами...")

    resolution = (100, 100, 100)
    scale = 25.0  # Уменьшен для более частых отверстий
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0
    threshold = 0.35  # Сlightly lowered
    num_voids = 60
    min_radius = 4
    max_radius = 10
    voxel_size = 0.5
    wall_thickness_voxels = 4  # Толщина стенки в вокселях (4 voxels * 0.5mm = 2mm)

    mask = create_perlin_noise_shape(
        resolution, scale, octaves, persistence, lacunarity, 
        threshold, wall_thickness_voxels
    )
    mask = create_spherical_voids(mask, num_voids, min_radius, max_radius, wall_thickness_voxels)
    
    # Сглаживание пещер
    mask = ndimage.binary_erosion(mask, iterations=1)

    mesh = mask_to_mesh(mask, voxel_size)
    mesh = cleanup_mesh(mesh)

    print(f"✅ Вершин: {len(mesh.vertices)}, Граней: {len(mesh.faces)}")
    print(f"📐 Объем: {mesh.volume:.2f} мм³")
    print(f"📏 Размеры: {mesh.extents}")

    mesh.export("cheese_structure.stl")
    print("💾 Файл сохранён: cheese_structure.stl")

if __name__ == "__main__":
    main()