from GameGenerator import Generator
from NN import TrainData

def main():
    game_generator = Generator(2)
    print(len(game_generator))
    game_generator.printGame(game_generator.getGameAtIndex(0))
    train_data = TrainData(game_generator.completed_games)




if __name__ == '__main__':
    main()
