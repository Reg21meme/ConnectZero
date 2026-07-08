import argparse
from connectzero.env.connect4 import Connect4
from connectzero.env.baselines import RandomAgent, HeuristicAgent


def get_agent(name):
    if name == "random":
        return RandomAgent()
    elif name == "heuristic":
        return HeuristicAgent()
    else:
        raise ValueError(f"Unknown agent: {name}. Choose from: random, heuristic")


def play_game(agent1, agent2, render=True):
    g = Connect4()
    agents = {1: agent1, 2: agent2}

    if render:
        print("\n=== ConnectZero ===")
        g.render()

    while not g.done:
        current = g.current_player
        agent = agents[current]
        col = agent.select_move(g.board, g.legal_moves(), current)

        if render:
            print(f"Player {current} ({'X' if current == 1 else 'O'}) plays column {col}")

        _, reward, done, info = g.step(col)

        if render:
            g.render()

    if render:
        if info["winner"] is None:
            print("=== Draw! ===")
        else:
            symbol = "X" if info["winner"] == 1 else "O"
            print(f"=== Player {info['winner']} ({symbol}) wins! ===")

    return info["winner"]


def cmd_play(args):
    if args.agent == "vs":
        agent1 = get_agent("random")
        agent2 = get_agent("heuristic")
        print("Random (X) vs Heuristic (O)")
    else:
        agent1 = get_agent(args.agent)
        agent2 = get_agent("random")
        print(f"{args.agent.capitalize()} (X) vs Random (O)")

    play_game(agent1, agent2, render=True)


def cmd_train(args):
    import yaml
    from connectzero.train.train_loop import train

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Starting training with config: {args.config}")
    train(**config)


def main():
    parser = argparse.ArgumentParser(prog="connectzero")
    subparsers = parser.add_subparsers(dest="command")

    play_parser = subparsers.add_parser("play", help="Play a game between two agents")
    play_parser.add_argument(
        "--agent",
        type=str,
        default="random",
        choices=["random", "heuristic", "vs"],
    )

    train_parser = subparsers.add_parser("train", help="Run training loop")
    train_parser.add_argument(
        "--config",
        type=str,
        default="configs/local_cpu.yaml",
        help="Path to training config yaml",
    )

    args = parser.parse_args()

    if args.command == "play":
        cmd_play(args)
    elif args.command == "train":
        cmd_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()