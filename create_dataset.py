import random
import pandas as pd
from datasets import Dataset, load_dataset
from game_of_life import GameOfLife

#TODO: include the empty game board

def create_random_problem_solution_pair(size=(5, 5)):
    density = random.randint(2, 4) * 0.1
    game = GameOfLife(size=size)
    game.random_board(density=density)
    game.next() #just to make sure its valid
    problem = str(game)
    explanation = game.next_with_explanation()
    solution = str(game)
    return problem, solution, explanation


def test_load_dataset():
    dataset = load_dataset("parquet", data_files="data/game_of_life.parquet")["train"]
    print(dataset[0]["question"]) #train is automatically created
    exit()

if __name__ == "__main__":
    test_load_dataset()
    SIZE = 50000
    questions = []
    answers = []

    print("Generating dataset...")

    for i in range(SIZE):
        problem, solution, explanation = create_random_problem_solution_pair()
        questions.append(problem)
        answers.append(explanation)
    
    print("Saving dataset...")

    data = {
        "question": questions,
        "answer": answers
    }

    dataset = Dataset.from_dict(data)
    dataset.to_parquet("data/game_of_life.parquet")
    print("Done!")
    '''
    problem, solution, explanation = create_random_problem_solution_pair()
    print("Problem:")
    print(problem)
    print("\nExplanation:")
    print(explanation)
    print("\nSolution:")
    print(solution)
    '''