#recrate rock paper scissors game using object oriented programming

import random




class RPS5Game:
    def __init__(self):
        self.user_score = 0
        self.computer_score = 0
        self.ties = 0

    def play_round(self, user_choice, computer_choice):
        if user_choice == computer_choice:
            self.ties += 1
            print("Tie")
            return "Tie" , computer_choice
        elif user_choice == "rock":
            if computer_choice == "paper" or computer_choice=="donut":
                self.computer_score += 1
                print("You lose.")
                return "You lose." , computer_choice
            else:
                self.user_score += 1
                print("You win!")
                return "You win!", computer_choice
        elif user_choice == "paper":
            if computer_choice == "scissors" or computer_choice=="gun":
                self.computer_score += 1
                print("You lose.")
                return "You lose." , computer_choice
            else:
                self.user_score += 1
                print("You win!")
                return "You win!" , computer_choice
            
        elif user_choice == "scissors":
            if computer_choice == "rock" or computer_choice=="gun":
                self.computer_score += 1
                print("You lose.")
                return "You lose." , computer_choice
            else:
                self.user_score += 1
                print("You win!")
                return "You win!" , computer_choice
        elif user_choice == "gun":
            if computer_choice == "rock" or computer_choice=="donut":
                self.computer_score += 1
                print("You lose.")
                return "You lose." , computer_choice
            else:
                self.user_score += 1
                print("You win!")
                return "You win!" , computer_choice
        elif user_choice == "donut":
            if computer_choice == "paper" or computer_choice=="scissors":
                self.computer_score += 1
                print("You lose.")
                return "You lose." , computer_choice
            else:
                self.user_score += 1
                print("You win!")
                return "You win!" , computer_choice


        

    def get_score(self):
        return self.user_score, self.computer_score, self.ties
    
    def reset_score(self):
        self.user_score = 0
        self.computer_score = 0
        self.ties = 0

    def get_computer_choice(self):
        return random.choice(["rock", "paper", "scissors","gun","donut"])

