import simulation_env as simulation

def main():
    env = simulation.env(render_mode="human")

    env.reset(seed=42)

    turns = 1

    for agent in env.agent_iter():
        #print("agent:", agent)
        observation, reward, termination, truncation, info = env.last()
        #print("observation:")
        #print(observation)
        #print("reward:")
        #print(reward)
        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()
        print(action)

        env.step(action)
        env.render()
        


    env.close()


if __name__ == "__main__":
    main()