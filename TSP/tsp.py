
import turtle
import math
import random

#Config
N_POP = 128
GEN_LIMIT = 5000
START_COORD = (-380.0, -350.0)
FILE_NAME = "cities100.txt"
PX = 0.9  # crossover prob
random.seed(42)  # reproducibility

#Data I/O 
def load_cities(path):
    cities = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                cities.append((x, y))
    return cities

#Geometry
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def route_distance(route, cities):
    """Distance from START -> route[0] -> ... -> route[-1] -> START"""
    total = dist(START_COORD, cities[route[0]])
    for i in range(len(route) - 1):
        total += dist(cities[route[i]], cities[route[i + 1]])
    total += dist(cities[route[-1]], START_COORD)
    return total

#GA Operators
def rank_selection(pop, fitnesses):
    """
    Rank-based selection (linear): best has highest weight.
    We sort by fitness ascending (lower is better).
    Weights = 1..N (worst..best), then roulette on those weights.
    """
    N = len(pop)
    idx_sorted = sorted(range(N), key=lambda i: fitnesses[i])  # best first
    weights = list(range(1, N + 1))  # 1..N
    weights.reverse()  # best gets N
    total_w = sum(weights)
    r = random.uniform(0, total_w)
    acc = 0.0
    for rank_pos, idx in enumerate(idx_sorted):
        acc += weights[rank_pos]
        if acc >= r:
            return pop[idx]
    return pop[idx_sorted[-1]]

def successor_in_parent(parent, city):
    """Return the successor of 'city' in 'parent' treating parent as a cycle."""
    n = len(parent)
    i = parent.index(city)
    return parent[(i + 1) % n]

def heuristic_crossover(p1, p2, cities):
    """
    Heuristic (greedy) crossover for TSP:
    - Start from a random city (from either parent's first gene).
    - At each step, consider the two parent successors of the current city.
      Pick the closer unused successor; if neither is available, pick the nearest unused city overall.
    Produces a valid permutation.
    """
    n = len(p1)     # Start from first gene of either parent (random choice)

    current = random.choice([p1[0], p2[0]])
    child = [current]
    used = {current}

    while len(child) < n: # Candidate successors from both parents
        
        cand = []
        s1 = successor_in_parent(p1, current)
        s2 = successor_in_parent(p2, current)
        if s1 not in used:
            cand.append(s1)
        if s2 not in used and s2 != s1:
            cand.append(s2)

        if cand:
            # choose the one closer to current
            best = None
            best_d = None
            for c in cand:
                d = dist(cities[current], cities[c])
                if best is None or d < best_d:
                    best, best_d = c, d
            nxt = best
        else:
            # fallback: nearest unused city overall
            best = None
            best_d = None
            for c in range(n):
                if c in used:
                    continue
                d = dist(cities[current], cities[c])
                if best is None or d < best_d:
                    best, best_d = c, d
            nxt = best

        child.append(nxt)
        used.add(nxt)
        current = nxt

    return child

def flip_bit_mutation(ind):
    """
    Flip-bit mutation for TSP (permutation-safe):
    randomly select two cut points and reverse the segment (bit-flip equivalent).
    """
    n = len(ind)
    a, b = sorted(random.sample(range(n), 2))
    ind[a:b + 1] = reversed(ind[a:b + 1])

#Drawing
def draw_route(route, cities):
    screen = turtle.Screen()
    screen.title("TSP Best Route (GA)")
    # Set a reasonable world coords span; assumes coordinates are roughly in [-400, -400]..[400, 400]
    margin = 40
    xs = [START_COORD[0]] + [c[0] for c in cities] + [START_COORD[0]]
    ys = [START_COORD[1]] + [c[1] for c in cities] + [START_COORD[1]]
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin
    screen.setworldcoordinates(min_x, min_y, max_x, max_y)

    pen = turtle.Turtle(visible=False)
    pen.speed(0)
    pen.penup()

    # Draw cities
    pen.color("black")
    for (x, y) in cities:
        pen.goto(x, y)
        pen.dot(5)

    # Draw start point
    pen.goto(START_COORD[0], START_COORD[1])
    pen.dot(10, "red")

    # Draw route path
    pen.color("blue")
    pen.pensize(2)
    pen.penup()
    pen.goto(START_COORD[0], START_COORD[1])
    pen.pendown()
    for idx in route:
        pen.goto(cities[idx][0], cities[idx][1])
    pen.goto(START_COORD[0], START_COORD[1])

    # Keep window open until click
    screen.exitonclick()

#GA Loop
def main():
    cities = load_cities(FILE_NAME)
    n = len(cities)
    if n == 0:
        print("No cities loaded. Check file format.")
        return

    PM = max(1.0 / n, 0.01)  # mutation probability per child (simple heuristic)

    # Initial population: random permutations
    population = [random.sample(range(n), n) for _ in range(N_POP)]

    # Track global bests
    best_two = []  # list of tuples (dist, route)

    for gen in range(GEN_LIMIT):
        fitnesses = [route_distance(ind, cities) for ind in population]

        # Update best two (unique by permutation identity)
        paired = sorted(zip(fitnesses, population), key=lambda t: t[0])
        for d, r in paired[:5]:  # check top few
            if not any(r == rr for _, rr in best_two):
                best_two.append((d, r[:]))
        best_two = sorted(best_two, key=lambda t: t[0])[:2]

        # Elitism: carry the best 2
        elites = [paired[0][1][:], paired[1][1][:]]

        # New generation
        new_pop = elites[:]
        while len(new_pop) < N_POP:
            p1 = rank_selection(population, fitnesses)
            p2 = rank_selection(population, fitnesses)
            if random.random() < PX:
                child = heuristic_crossover(p1, p2, cities)
            else:
                # copy one parent
                child = p1[:]
            if random.random() < PM:
                flip_bit_mutation(child)
            new_pop.append(child)

        population = new_pop

        # Optional progress print
        if (gen + 1) % 1000 == 0:
            print(f"Generation {gen + 1}/{GEN_LIMIT} | Best sum of distance: {best_two[0][0]:.3f}")

    # Final evaluation to ensure best_two is up to date
    fitnesses = [route_distance(ind, cities) for ind in population]
    paired = sorted(zip(fitnesses, population), key=lambda t: t[0])
    for d, r in paired[:5]:
        if not any(r == rr for _, rr in best_two):
            best_two.append((d, r[:]))
    best_two = sorted(best_two, key=lambda t: t[0])[:2]

    #Print results
    print("\n=== Best Two Results ===")
    for i, (d, r) in enumerate(best_two, start=1):
        print(f"[{i}] Sum of distances = {d:.6f}")
        print(f"    Individual (permutation) = {r}")

    #Draw best route
    best_route = best_two[0][1]
    draw_route(best_route, cities)

if __name__ == "__main__":
    main()
