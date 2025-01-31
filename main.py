import simulation_env as simulation

def main():
    env = simulation.env(render_mode="human")

    env.reset(seed=42)

    turns = 10

    for _ in range(turns):
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                # this is where you would insert your policy
                action = env.action_space(agent).sample()

            env.step(action)


    env.close()


if __name__ == "__main__":
    main()