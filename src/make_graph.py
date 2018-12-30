import matplotlib.pyplot as plt
from decimal import Decimal

train_timesteps = []
train_results = []
eval_timesteps = []
eval_results = []
train_obs_skip = 100
with open("train_results_a2c.txt", "r+") as f:
    mean_res = 0.0
    i = 0
    for line in f.readlines():
        timestep, result = line.split(",")
        mean_res += float(result)
        if i % train_obs_skip == 0 and i:
            timestep = '%.2E' % Decimal(timestep)
            train_timesteps.append(timestep)
            mean_res /= train_obs_skip
            train_results.append(round(mean_res, 2))
            mean_res = 0
        i += 1

with open("eval_results_a2c.txt", "r+") as f:
    for line in f.readlines():
        timestep, result = line.split(",")

        timestep = '%.2E' % Decimal(timestep)
        result = round(float(result), 2)
        eval_timesteps.append(timestep)
        eval_results.append(result)

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(train_timesteps, train_results, label="Train")
ax.plot(train_timesteps, eval_results, label="Eval")
plt.legend(fontsize="x-large")
plt.xticks(rotation=90)
ax.set_xticks(ax.get_xticks()[::10])
ax.set_xlabel("Timesteps")
plt.gcf().subplots_adjust(bottom=0.17)
ax.set_ylabel("Reward")
fig.suptitle("Results")
fig.savefig("results/results.png")
plt.show()
