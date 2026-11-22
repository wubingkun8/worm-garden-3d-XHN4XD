# Block 1: world geometry and camera conversion
import pygame
import numpy as np
import math

VERTS = np.array([
    [-.5,-.5,-.5], [.5,-.5,-.5], [.5,.5,-.5], [-.5,.5,-.5],
    [-.5,-.5,.5],  [.5,-.5,.5],  [.5,.5,.5],  [-.5,.5,.5]
], dtype=float)
EDGES = [(0,1),(1,2),(2,3),(3,0),
         (4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]

def unit(v):
    return v / np.linalg.norm(v)

def to_camera(point, eye, target):
    f = unit(target-eye)
    # right = forward x world_up, so east stays on the right like in the 2D game
    r = unit(np.cross(f, np.array([0., 1., 0.])))
    u = np.cross(r, f)
    d = point-eye
    return np.array([np.dot(d,r), np.dot(d,u), np.dot(d,f)])
# Block 2: switchable projection
def project(p, perspective, width=800, height=600):
    x, y, z = p
    if z <= 0.1:
        return None
    if perspective:
        f = (height/2)/math.tan(math.radians(65)/2)
        return (int(width/2 + f*x/z), int(height/2 - f*y/z))
    return (int(width/2 + 140*x), int(height/2 - 140*y))
# Block 3: orbiting wireframe cube and P-key toggle
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
perspective = True
angle = 0.0
running = True

while running:
    dt = min(clock.tick(60)/1000, 0.05)
    angle += 0.5*dt
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            perspective = not perspective

    eye = np.array([3*math.cos(angle), 2.0,
                    3*math.sin(angle)], dtype=float)
    target = np.array([0.,0.,0.])
    points = [project(to_camera(v,eye,target), perspective)
              for v in VERTS]
    screen.fill((20, 27, 36))
    for i,j in EDGES:
        if points[i] is not None and points[j] is not None:
            pygame.draw.line(screen,(155,235,180),points[i],points[j],2)
    pygame.display.flip()
pygame.quit()
