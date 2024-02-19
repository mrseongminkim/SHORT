import re
import matplotlib.pyplot as plt

# Define regex patterns
iter_pattern = re.compile(r'Iter (\d+)')
valid_loss_pattern = re.compile(r'valid_loss: ([\d.]+)e[+\-](\d+)')

# Initialize lists to store values
iters = []
valid_losses = []

# Read the text file
with open('one-hop copy.txt', 'r') as file:
    lines = file.readlines()

# Parse the values from the text file
for line in lines:
    iter_match = iter_pattern.search(line)
    valid_loss_match = valid_loss_pattern.search(line)

    if iter_match:
        iter_value = int(iter_match.group(1))
        iters.append(iter_value)

    if valid_loss_match:
        loss_value = float(valid_loss_match.group(1)) * 10 ** int(valid_loss_match.group(2))
        valid_losses.append(loss_value)

# Plot the values
plt.plot(iters, valid_losses)
plt.xlabel('Iteration')
plt.ylabel('Valid Loss')
plt.title('Validation Loss over Iterations')
plt.show()
fig = plt.gcf()
fig.set_size_inches(6, 3)
fig.savefig('data_size.pdf', bbox_inches='tight')