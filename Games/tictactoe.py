def display_board(board):
	print()
	print('   |   |')
	print(' ' + board[7] + ' | ' + board[8] + ' | ' + board[9])
	print('   |   |')
	print('-----------')
	print('   |   |')
	print(' ' + board[4] + ' | ' + board[5] + ' | ' + board[6])
	print('   |   |')
	print('-----------')
	print('   |   |')
	print(' ' + board[1] + ' | ' + board[2] + ' | ' + board[3])
	print('   |   |', '\n')

def player_input():

	marker = ''
	options = ['X', 'O']

	# ASSIGN X OR O TO PLAYER 1

	while marker not in options:
		marker = input('Player 1, choose X or O : ')

	player1 = marker
	if player1 == 'X':
		player2 = 'O'
	else:
		player2 = 'X'

	return player1, player2

def place_marker(board, marker, position):
	board[position] = marker
	return board

def win_check(board, mark):

	# CHECK FOR WIN
	for i in range(1,4):
		if board[i] == board[i+3] == board[i+6] == mark:
			return True
	for i in range(1,8,3):
		if board[i] == board[i+1] == board[i+2] == mark:
			return True
	if board[1] == board[5] == board[9] == mark:
		return True
	if board[3] == board[5] == board[7] == mark:
		return True

	return False

def space_check(board, position):
    # TRUE IF THERE IS A FREE SPACE
	return board[position] == ' '

def full_board_check(board):
    # TRUE IF BOARD IS FULL

    for i in range(1,10):
    	if space_check(board, i):
    		return False

    return True

def player_choice(board):

	# INITIAL
	choice = 'WRONG'
	acceptable_input = range(1,10)
	acceptable_number = False

	# CHECK FOR DIGIT AND RANGE
	while not ( choice.isdigit() and acceptable_number ) :   

		choice = input("Choose an index position (1 - 9) : ")

		# DIGIT CHECK
		if choice.isdigit():
			# RANGE CHECK
			if int(choice) in acceptable_input:
				if not space_check(board, int(choice)):
					print('Please choose a free space.')
				else:	acceptable_number = True
			else:
				print('Input was out of acceptable range (1 - 9)')
		else:
			print('Input was not a digit!')

	return int(choice)

def replay():
    
    key = False

    while not key:

    	ans = input('Do you want to play again? Y or N : ')

    	if ans == 'Y':
    		play_again = True
    		key = True
    	elif ans == 'N':
    		play_again = False
    		key = True

    return play_again

print('\nWelcome to Tic Tac Toe!\n')
print('Here is a guide for the index positions : ')
display_board(['#','1','2','3','4','5','6','7','8','9'])

while True:

	# INITIALISE

	board = ['#',' ',' ',' ',' ',' ',' ',' ',' ',' ']
	player1_marker, player2_marker = player_input()
	display_board(board)
	game_on = True

	while game_on:

		# Player 1 Turn

		print('Player 1 : ')
		player1_choice = player_choice(board)
		board = place_marker(board, player1_marker, player1_choice)
		display_board(board)

		# Check if Player 1 has won
		if win_check(board, player1_marker):
			print('Player 1 is the WINNER!')
			break
		# Check for draw
		if full_board_check(board):
			print('The game is a DRAW!')
			break

		# Player 2 turn

		print('Player 2 : ')
		player2_choice = player_choice(board)
		board = place_marker(board, player2_marker, player2_choice)
		display_board(board)

		# Check if Player 2 has won
		if win_check(board, player2_marker):
			print('Player 2 is the WINNER!')
			break
		# Check for draw
		if full_board_check(board):
			print('The game is a DRAW!')
			break

	if not replay():
		break



