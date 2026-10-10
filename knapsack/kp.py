
import random
import math


N_POP     = 128
GEN_LIMIT = 3000
FILE_NAME = "kp_dataset(100).txt"

TOURN_K   = 3
PX        = 0.9
PROGRESS_EVERY = 100
# random.seed(42)


def load_dataset(path):
    """Load dataset: each line -> weight value"""
    W, V = [], []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    capacity = 5000  # fixed for this dataset
    for ln in lines:
        ln = ln.replace(",", " ")
        parts = ln.split()
        if len(parts) >= 2:
            a, b = parts[:2]
            W.append(int(a))
            V.append(int(b))
    if not W:
        raise ValueError("No items parsed from dataset. Check file format.")
    return capacity, W, V

def rand_individual(n):
    return [random.randint(0, 1) for _ in range(n)]

def totals(ind, W, V):
    w = sum(wi for wi, b in zip(W, ind) if b)
    v = sum(vi for vi, b in zip(V, ind) if b)
    return w, v

# Repair Operator
def repair(ind, capacity, W, V):
    """Ensure individual is feasible by removing least valuable items."""
    w, _ = totals(ind, W, V)
    if w <= capacity:
        return ind
    items = [(i, V[i]/W[i] if W[i] > 0 else 0) for i in range(len(W)) if ind[i] == 1]
    for i, _ in sorted(items, key=lambda x: x[1]):  # drop lowest ratio
        ind[i] = 0
        w, _ = totals(ind, W, V)
        if w <= capacity:
            break
    return ind

def fitness(ind, capacity, W, V):
    w, v = totals(ind, W, V)
    if w > capacity:
        return 0
    return v

def tournament_select(pop, fits, k=TOURN_K):
    best = None
    for _ in range(k):
        i = random.randrange(len(pop))
        if best is None or fits[i] > fits[best]:
            best = i
    return pop[best][:]

# Two-Point Crossover
def two_point_crossover(a, b, px=PX):
    n = len(a)
    if random.random() > px or n < 3:
        return a[:], b[:]
    p1, p2 = sorted(random.sample(range(1, n - 1), 2))
    c1 = a[:p1] + b[p1:p2] + a[p2:]
    c2 = b[:p1] + a[p1:p2] + b[p2:]
    return c1, c2

# Adaptive Mutation
def adaptive_mutation_rate(gen, gen_limit, base_rate):
    return base_rate * (1.5 - 0.5 * gen / gen_limit)

def mutate(ind, pm):
    for i in range(len(ind)):
        if random.random() < pm:
            ind[i] ^= 1

# Diversity Metric
def population_diversity(pop):
    n = len(pop)
    genome_len = len(pop[0])
    bit_sums = [sum(ind[i] for ind in pop) / n for i in range(genome_len)]
    entropy = -sum(
        p * math.log(p + 1e-9, 2) + (1 - p) * math.log(1 - p + 1e-9, 2)
        for p in bit_sums
    ) / genome_len
    return entropy  # 0–1, higher = more diverse

# Main Evolution Loop
def run():
    capacity, W, V = load_dataset(FILE_NAME)
    n_items = len(W)
    base_PM = 1.0 / n_items

    population = [rand_individual(n_items) for _ in range(N_POP)]

    # Track best two overall across all generations
    best1_ever, best2_ever = None, None
    best1_fit, best2_fit = -1, -1

    print(f"[INFO] items={n_items}, capacity={capacity}")
    print(f"[INFO] population={N_POP}, generations={GEN_LIMIT}")

    for gen in range(1, GEN_LIMIT + 1):
        
        population = [repair(ind, capacity, W, V) for ind in population] # Evaluate and repair
        fits = [fitness(ind, capacity, W, V) for ind in population]

        
        order = sorted(range(len(population)), key=lambda i: fits[i], reverse=True) # Rank current generation
        top1, top2 = order[0], order[1]

        
        if fits[top1] > best1_fit: # Update global top 2
            best2_fit, best2_ever = best1_fit, best1_ever
            best1_fit, best1_ever = fits[top1], population[top1][:]
        elif fits[top1] > best2_fit and (best1_ever is None or population[top1] != best1_ever):
            best2_fit, best2_ever = fits[top1], population[top1][:]

  
        PM = adaptive_mutation_rate(gen, GEN_LIMIT, base_PM)       # Adaptive mutation rate


        diversity = population_diversity(population)         # Maintain diversity
        if diversity < 0.25:
            for _ in range(int(0.1 * N_POP)):
                population[random.randrange(N_POP)] = rand_individual(n_items)


        if gen == 1 or gen % PROGRESS_EVERY == 0 or gen == GEN_LIMIT:         # Print progress
            w1, v1 = totals(population[top1], W, V)
            print(f"Gen {gen:4d} ->  Max sum of values : {v1:>6d}")



        new_pop = [population[top1][:], population[top2][:]]         # Elitism + offspring generation
        while len(new_pop) < N_POP:
            p1 = tournament_select(population, fits)
            p2 = tournament_select(population, fits)
            c1, c2 = two_point_crossover(p1, p2)
            mutate(c1, PM)
            mutate(c2, PM)
            new_pop.append(repair(c1, capacity, W, V))
            if len(new_pop) < N_POP:
                new_pop.append(repair(c2, capacity, W, V))
        population = new_pop

    # Final Global Top-2 Results
    print("\n=== BEST TWO INDIVIDUALS (OVERALL) ===")
    for rank, (ind, fit) in enumerate([(best1_ever, best1_fit), (best2_ever, best2_fit)], start=1):
        if ind is None: 
            continue
        w, v = totals(ind, W, V)
        bits = "".join(map(str, ind))
        print(f"{rank} Value: {v}   Weight: {w} ")
        print(f"    {bits[:120]}{'...' if len(bits)>120 else ''}")

if __name__ == "__main__":
    run()
