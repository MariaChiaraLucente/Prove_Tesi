import pygame
import game_state
from config import *

class Paddle:
    def __init__(self, side):
        self.width = 20
        self.height = 140
        self.x = 50 if side == "player1" else SCREEN_WIDTH - 70
        self.rect = pygame.Rect(self.x, SCREEN_HEIGHT // 2, self.width, self.height)

    def update(self, yolo_y):
        # Mappiamo la coordinata Y dalla camera (480) allo schermo (720)
        target_y = (yolo_y / CAM_HEIGHT) * SCREEN_HEIGHT
        # Interpolazione lineare per un movimento fluido (smoothing)
        self.rect.centery += (target_y - self.rect.centery) * 0.2


    class Ball:
        def __init__(self):
            self.pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]
            self.radius = 15

        def update(self, yolo_coords, ball_out):
            if yolo_coords:
                # Mappiamo le coordinate X e Y della pallina fisica rilevata
                self.pos[0] = (yolo_coords[0] / CAM_WIDTH) * SCREEN_WIDTH
                self.pos[1] = (yolo_coords[1] / CAM_HEIGHT) * SCREEN_HEIGHT
                if len(yolo_coords) >= 3:
                    self.radius = yolo_coords[2]

                # Se le coordinate del box vanno oltre le pareti di destra e sinistra, il gioco si ferma
                # la palla va al centro. e spunta il messaggio "palla al centro, pronto a ripartire"
                if ball_out:
                    self.pos[0] = SCREEN_WIDTH // 2
                    self.pos[1] = SCREEN_HEIGHT // 2
                    game_state.paused = True
                    print("[INFO] Palla al centro, pronto a ripartire")

