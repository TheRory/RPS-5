from tkinter import *
from tkinter import font
from PIL import Image, ImageTk
import cv2 as cv
import rps5
import FINALreloadmodel as fmrg
import os


root = Tk()
root.title('RPS5')
screen_width=int(800*1.5)
screen_height=int(500*1.5)
root.minsize(screen_width, screen_height)
root.maxsize(screen_width, screen_height)
root.configure(bg='#58F')
game=rps5.RPS5Game()
load_dir = os.path.join(os.path.dirname(__file__), 'gui images')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Unable to read camera feed")

# Set the desired width and height for the camera feed display
feed_width = 450
feed_height = 300

def playround():
    image = Image.fromarray(img1)
    result=game.play_round(fmrg.predict(fmrg.load_model('finalmodelv3.3.pth'),fmrg.imageloaded(image)),game.get_computer_choice())
    opponentimg(result[1])
    if result[0]=='You win!':
        f1.configure(bg='green')
        oponentf1.configure(bg='red')
    elif result[0]=='You lose.':
        f1.configure(bg='red')
        oponentf1.configure(bg='green')
    else:
        f1.configure(bg='white')
        oponentf1.configure(bg='white')
    update_score_labels()
        
def opponentimg(r):
    global opmoveimg  # Use the global opmoveimg
    opmoveimg = Image.open(load_dir + '/' + r + '.png')
    opmoveimg = opmoveimg.resize((feed_width, feed_height), Image.Resampling.LANCZOS)
    opmoveimg = ImageTk.PhotoImage(opmoveimg)
    oponentl1['image'] = opmoveimg

def exitWindow():
    cap.release()
    cv.destroyAllWindows()
    root.destroy()
    root.quit()

def reset_score():
    game.reset_score()
    update_score_labels()

def update_score_labels():
    player_score, computer_score, ties = game.get_score()
    player_score_label.config(text=f'Player: {player_score}')
    computer_score_label.config(text=f'Chance: {computer_score}')
    ties_label.config(text=f'Ties: {ties}')
    '''global opmoveimg  # Use the global opmoveimg
    opmoveimg = Image.open('C:/Users/roryu/Desktop/rpsPytorch/gui images/' + game.get_computer_choice() + '.png')
    opmoveimg = opmoveimg.resize((feed_width, feed_height), Image.ANTIALIAS)
    opmoveimg = ImageTk.PhotoImage(opmoveimg)
    oponentl1['image'] = opmoveimg'''
    
    root.after(1000, update_score_labels)

f1 = LabelFrame(root, bg='white', bd=12)
#put f1 to the top left of the screen
f1.place(x=30, y=30)
l1 = Label(f1, bg='white')
l1.pack()
oponentf1=LabelFrame(root, bg='white', bd=12)
oponentf1.place(x=screen_width - feed_width-50, y=30)
image = Image.open('C:/Users/roryu/Desktop/rpsPytorch/robot.png')
image= image.resize((feed_width, feed_height), Image.Resampling.LANCZOS)
image = ImageTk.PhotoImage(image)
oponentl1 = Label(oponentf1, image=image, bg='white')
oponentl1.image = image  # Important to prevent the image from being garbage collected
oponentl1.pack()

#put player score and computer score 20 pixels above the bottom of the screen, find scores using game.getscore()

custom_font = font.Font(family='Helvetica', size=16, weight='bold', slant='italic')
computer_score_label = Label(root, width=20, height=10,text='Chance: 0', bg='#58F', fg='white',font=custom_font)  # Adjust the text and size as needed

# Place the label in the middle horizontally and at the bottom vertically
computer_score_label.place(relx=0.5, rely=0.95, anchor='s')

player_score_label = Label(root, width=20, height=10,text='Player: 0', bg='#58F', fg='white', font=custom_font)  # Adjust the text and size as needed
player_score_label.place(relx=0.3, rely=0.95, anchor='s')

ties_label = Label(root, width=20, height=10,text='Ties: 0', bg='#58F', fg='white',font=custom_font)  # Adjust the text and size as needed
ties_label.place(relx=0.7, rely=0.95, anchor='s')

'''player_score_label = Label(root, text='Player: 0', bg='blue', fg='white')
player_score_label.pack(side=BOTTOM, padx=20)
computer_score_label = Label(root, text='Chance: 0', bg='blue', fg='white')
player_score_label.pack(side=BOTTOM, padx=30)
ties_label = Label(root, text='Ties: 0', bg='blue', fg='white')
player_score_label.pack(side=BOTTOM, padx=40)
'''

b1 = Button(root, bg='green', fg='white', activebackground='white', activeforeground='green', text='Play Round',
            relief=RIDGE, height=5, width=10, command=playround)
#put b1 30 pixels below f1
b1.place(x=30, y=feed_height+60)

b2 = Button(root, fg='white', bg='red', activebackground='white', activeforeground='red', text='EXIT',
            relief=RIDGE, height=5, width=10, command=exitWindow)
#put b1 30 pixels below oponentf1
b2.place(x=screen_width -100, y=feed_height+60)

b3 = Button(root, fg='white', bg='blue', activebackground='white', activeforeground='blue', text='Reset Score',
            relief=RIDGE, height=5, width=10, command=reset_score)
#put b1 30 pixels below oponentf1
b3.place(x=screen_width/2-50, y=feed_height+60)




while True:
    img = cap.read()[1]
    img = cv.flip(img, 1)
    
    # Resize the frame to the desired dimensions
    img = cv.resize(img, (feed_width, feed_height))
    # put frame on the left side of the screen
    img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # put frame on the left side of the screen
    


    img = ImageTk.PhotoImage(Image.fromarray(img1))
    l1['image'] = img
    root.update()

cap.release()