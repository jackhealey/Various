# Scrabble Game Python Project

# Seperate the letters into groups in order to achieve a variety of letters in the randomly generated rack

import random
import sys
LetterVowels = ["a","e","i","o","u"]
LetterCommonConsonants = ["g","d","l","s","n","r","t","b","c","m","p"]
LetterRareConsonants = ["z","q","j","x","k","v","w","y"]
SowPods = []

# Import dictionary words in order to later compare to input word

myFile = open("sowpods.txt","r")
for Line in myFile:
    SowPods.append(Line.strip())
myFile.close()

# Assign score values to the letters mirroring Scrabble scoring with a dictionary

LetterValues = {"a": 1, "b": 3, "c": 3, "d": 2, "e": 1, "f": 4, "g": 2, "h": 4, "i": 1, "j": 8, "k": 5, "l": 1, "m": 3, "n": 1, "o": 1, "p": 3, "q": 10, "r": 1, "s": 1, "t": 1, "u": 1, "v": 4, "w": 4, "x": 8, "y": 4, "z": 10}

# Provide instructions to play the game and information on the different scores for letters

print("Create a word from the randomly generated rack of letters to score points, different letters score different points, while longer words improve your score!\n")
print("a:1  b:3  c:3  d:2  e:1  f:4  g:2  h:4  i:1  j:8  k:5  l:1  m:3  n:1  o:1  p:3  q:10  r:1  s:1  t:1  u:1  v:4  w:4  x:8  y:4  z:10\n")
print("The rack:")

# Generate the rack using mostly vowels and common consonants mirroring the distribution in Scrabble

SelectedLetters = []
for i in range(3):
    SelectedLetters.append(random.choice(LetterVowels))
    SelectedLetters.append(random.choice(LetterCommonConsonants))
    SelectedLetters.append(random.choice(LetterVowels))
    SelectedLetters.append(random.choice(LetterCommonConsonants))
    SelectedLetters.append(random.choice(LetterRareConsonants))
print(SelectedLetters)

# Set out the rules of the game clearly

print("\nTips: Generated letters can be used more than once! \n      If you use a letter not in the rack you get zero points!")

# Define a function which adds the score for each letter in the inputted word

def ScrabbleScore(InputWord):
    Score = 0
    for i in InputWord:
        Score = Score+LetterValues[i.lower()]
    return Score

# Set up an input for the found word 

InputWord = input("\nCreate a word from the rack above and press enter: ")

# Limit the input word so that it can only contain letters from the rack


for letter in InputWord:
    if letter not in SelectedLetters:
        print("\nYou have to use letters in the rack... Zero points!")
        sys.exit()

# If the word is verified by the dictionary (SOWPODS), display points gained

# If the input is not a word, the player gets a second chance, then another fail means zero points gained

# The player gets a reaction depending on the score achieved

if InputWord.upper() in SowPods:
    print (str(ScrabbleScore(InputWord)) + " points!")
    if 1 <= (ScrabbleScore(InputWord)) <= 5:
        print("\nYou should have another go...")
    elif 6 <= (ScrabbleScore(InputWord)) <= 10:
        print("\nYou can do better...")
    elif 11 <= (ScrabbleScore(InputWord)) <= 15:
        print("\nWell done!")
    elif 16 <= (ScrabbleScore(InputWord)) <= 20:
        print("\nGreat score!")
    elif (ScrabbleScore(InputWord)) > 20:
        print("\nExcellent score!")
else:
    print ("That is not an acceptable word, try again!")
    InputWord = input("\nCreate a word from the rack above and press enter: ")
    for letter in InputWord:
        if letter not in SelectedLetters:
            print("\nYou have to use letters in the rack... Zero points!")
            sys.exit()
    if InputWord.upper() in SowPods:
        print (str(ScrabbleScore(InputWord)) + " points!")
        if 1 <= (ScrabbleScore(InputWord)) <= 5:
            print("\nYou should have another go...")
        elif 6 <= (ScrabbleScore(InputWord)) <= 10:
            print("\nYou can do better...")
        elif 11 <= (ScrabbleScore(InputWord)) <= 15:
            print("\nWell done!")
        elif 16 <= (ScrabbleScore(InputWord)) <= 20:
            print("\nGreat score!")
        elif (ScrabbleScore(InputWord)) > 20:
            print("\nExcellent score!")
    else:
        print ("\nYou haven't got the hang of this... Zero points!")

# Happy playing!
