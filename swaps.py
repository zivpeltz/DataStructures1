from AVLTree import AVLNode, AVLTree
import timeit
import matplotlib.pyplot as plt


def timed_insert(data):

    def f():
        t = AVLTree()
        for x in data:
            t.insert(x, x, start="max")

    # timeit.repeat gives you several runs so you can take the minimum or average
    times = timeit.repeat(f, number=1, repeat=5)
    return min(times)



n = 80_000
swaps = [1_000, 2_000, 4_000, 6_000, 8_000, 10_000, 12_000, 14_000, 16_000, 18_000, 20_000, 30_000, 35_000, 40_000, 50_000, 60_000, 70_000, 80_000]
# swaps = [k * 500 for k in range(100)]
t = []
for swap in swaps:
    swaped_part = list(range(swap))
    swaped_part.reverse()
    later = list(range(swap, n))

    data = swaped_part + later
    t.append(timed_insert(data))


# plot_multiple_lines([swaps], [t], "swaps")
plt.plot([0.5*s*(s-1) for s in swaps], t, marker='o', linestyle='-', color='g')  # Plot with dots and line
plt.title('Run time depending on the number of swaps for AVL Tree Insertion')
plt.xlabel('Number of Swaps (log)')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.show()