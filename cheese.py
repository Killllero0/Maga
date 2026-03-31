import numpy as np
import noise
import trimesh
from scipy import ndimage

def create_perlin_noise_shape(shape, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, threshold=0.3):
    """
    Создаёт 3D-маску на основе шума Перлина.
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

    mask = noise_map > threshold
    return mask

def create_spherical_voids(mask, num_voids=50, min_radius=3, max_radius=8):
    """
    Добавляет сферические пустоты в маску.
    """
    shape = mask.shape
    result = mask.copy()
    np.random.seed(42)

    for _ in range(num_voids):
        radius = np.random.randint(min_radius, max_radius + 1)
        x = np.random.randint(radius, shape[0] - radius)
        y = np.random.randint(radius, shape[1] - radius)
        z = np.random.randint(radius, shape[2] - radius)

        xx, yy, zz = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist = np.sqrt((xx - x)**2 + (yy - y)**2 + (zz - z)**2)
        sphere = dist <= radius
        result[sphere] = False

    return result

def mask_to_mesh(mask, voxel_size=1.0):
    """
    Конвертирует бинарную 3D-маску в меш.
    """
    from skimage import measure
    vertices, faces, normals, values = measure.marching_cubes(mask.astype(float), level=0.5)
    vertices *= voxel_size
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

def cleanup_mesh(mesh):
    """
    Очищает меш совместимым способом.
    """
    # Объединяем близкие вершины
    mesh.merge_vertices()
    
    # Удаляем вырожденные грани
    mesh.remove_degenerate_faces()
    
    # Удаляем бесконечные значения
    mesh.remove_infinite_values()
    
    # Удаляем непрочные грани
    if hasattr(mesh, 'remove_unreferenced_vertices'):
        mesh.remove_unreferenced_vertices()
    
    return mesh

def main():
    print("🔧 Генерация 3D-модели с шумом Перлина и сферическими пустотами...")

    resolution = (100, 100, 100)
    scale = 50.0
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0
    threshold = 0.2
    num_voids = 60
    min_radius = 4
    max_radius = 10
    voxel_size = 0.5

    mask = create_perlin_noise_shape(resolution, scale, octaves, persistence, lacunarity, threshold)
    mask = create_spherical_voids(mask, num_voids, min_radius, max_radius)
    mask = ndimage.binary_fill_holes(mask)
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