import torch
from queue import Queue

def retain_largest_island(image):
    r = len(image)
    c = len(image[0])
    
    ds = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    visited = [[0] * c for _ in range(r)]
    q = Queue()
    max_size = 0

    for i in range(r):
        for j in range(c):
            if image[i][j] == 1:
                indices = []
                size = 0
                q.put((i, j))
                visited[i][j] = 1
                while not q.empty():
                    x, y = q.get()
                    indices.append((x, y))
                    size += 1
                    for dx, dy in ds:
                        nx, ny = x + dx, y + dy
                        if nx >= 0 and nx < r and ny >= 0 and ny < c and image[nx][ny] == 1 and visited[nx][ny] == 0:
                            q.put((nx, ny))
                            visited[nx][ny] = 1
                max_size = max(max_size, size)
                if size == max_size:
                    for x, y in indices:
                        image[x][y] = size
    
# 示例用法
image = [
    [1, 1, 0, 0, 1],
    [1, 1, 0, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 0, 1, 0, 0]
]

retain_largest_island(image)

# 打印结果
for row in image:
    print(row)

image = torch.tensor(image)
image = (image == 5)
print(image)