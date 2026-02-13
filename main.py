from charset_normalizer import detect
import pygame
import cv2
import socket
from vision import VisionManager
from logic import Paddle
import game_state
from config import *

def main():
    # Inizializzazione Pygame e Socket
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Inizializzazione moduli
    vision = VisionManager()
    p1 = Paddle("player1")
    p2 = Paddle("player2")
    ball = Paddle.Ball()
    last_img_hands = None
    last_img_ball = None
    font = pygame.font.SysFont(None, 36)

    

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                game_state.paused = not game_state.paused


        if not game_state.paused:
            # 1. Rilevamento (Visione)
            img_hands, img_ball, detections = vision.update()

            if img_hands is None:
                continue
            last_img_hands = img_hands
            last_img_ball = img_ball

            # 2. Aggiornamento Logica (Mani -> Paddle, Forme -> Ball)
            hands = detections["hands"]

             # Controllo paddle sinistro (Player 1)
            if hands and len(hands) > 0 and hands[0] is not None:
               p1.update(hands[0][1])

            # Controllo paddle destro (Player 2)
            if hands and len(hands) > 1 and hands[1] is not None:
                p2.update(hands[1][1])  # Muove paddle player 2 con coordinata Y

            # Aggiorna posizione pallina fisica
            ball.update(detections["ball"], detections["ball_out"])

            # 3. Trasmissione UDP (opzionale, basato sul tuo codice precedente)
            if detections["ball"]:
                msg = f"BALL_X:{detections['ball'][0]}".encode()
                sock.sendto(msg, (UDP_IP, UDP_PORT))
        else:
            # Se il gioco è in pausa, mostriamo un messaggio al centro de
            # llo schermo
            pause_text = font.render("PAUSA - Porta il disco al centro e Premi 'E' per riprendere", True, (255, 0, 0))
            text_rect = pause_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT-30))
            screen.blit(pause_text, text_rect)
       

        # 4. Rendering (Grafica)
        screen.fill((200, 200, 200)) # Sfondo bianco
        
        # Disegno campo e oggetti
        pygame.draw.aaline(screen, (255, 255, 255), (SCREEN_WIDTH//2, 0), (SCREEN_WIDTH//2, SCREEN_HEIGHT))
        pygame.draw.rect(screen, (255, 255, 255), p1.rect)
        pygame.draw.rect(screen, (255, 255, 255), p2.rect)
        pygame.draw.circle(screen, (0, 255, 0), (int(ball.pos[0]), int(ball.pos[1])), ball.radius)
        
        # Anteprime Camera (Miniatura)
        if last_img_hands is not None and last_img_ball is not None:
            img_hands_rgb = cv2.cvtColor(last_img_hands, cv2.COLOR_BGR2RGB)
            img_ball_rgb = cv2.cvtColor(last_img_ball, cv2.COLOR_BGR2RGB)
            img_hands_small = cv2.resize(img_hands_rgb, (200, 160))
            img_ball_small = cv2.resize(img_ball_rgb, (160, 120))
            surf_hands = pygame.surfarray.make_surface(img_hands_small.swapaxes(0, 1))
            surf_ball = pygame.surfarray.make_surface(img_ball_small.swapaxes(0, 1))
            screen.blit(surf_hands, (10, 10))
            screen.blit(surf_ball, (180, 10))

        pygame.display.flip()


    pygame.quit()
    vision.cap.release()

if __name__ == "__main__":
    main()