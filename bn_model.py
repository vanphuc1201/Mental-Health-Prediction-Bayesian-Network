import pandas as pd
from pgmpy.models import BayesianNetwork, DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, f1_score
import networkx as nx
import matplotlib.pyplot as plt

# =========================
# 1. Load data
# =========================
train_df = pd.read_csv("features/bn_ground_truth_train.csv")
test_df = pd.read_csv("features/bn_ground_truth_test.csv")

# Force all columns to categorical (discrete)
train_df = train_df.astype("category")
test_df = test_df.astype("category")

# =========================
# 2. Structure Learning
# =========================
print("Learning structure...")

hc = HillClimbSearch(train_df)
best_dag = hc.estimate(scoring_method="bic-d")  # now pgmpy will treat data as discrete

print("Learned edges:")
print(best_dag.edges())

model = DiscreteBayesianNetwork(best_dag.edges())

# =========================
# 3. Parameter Learning
# =========================
print("Learning parameters...")

model.fit(
    train_df,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=10
)

# =========================
# 4. Inference Engine
# =========================
infer = VariableElimination(model)

# =========================
# 5. Prediction Function
# =========================
def predict(row):
    # Get all features except the target
    raw_evidence = row.drop("stress_level").to_dict()
    
    # FIX: Filter evidence to only include nodes that exist in the learned model
    evidence = {k: v for k, v in raw_evidence.items() if k in model.nodes()}
    
    result = infer.query(
        variables=["stress_level"],
        evidence=evidence
    )
    
    # result.values = [P(0), P(1)]
    return int(result.values[1] > result.values[0])


# =========================
# 6. Evaluate on test set
# =========================
y_true = test_df["stress_level"].astype(int).values  # convert to int for metrics
y_pred = []

for _, row in test_df.iterrows():
    y_pred.append(predict(row))

# =========================
# 7. Metrics
# =========================
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")

# 1. Setup the graph
G = nx.DiGraph(model.edges())

# 2. Define Node Colors
node_colors = []
for node in G.nodes():
    if node == 'stress_level':
        node_colors.append('#FFD700')  # Gold
    else:
        node_colors.append('#A0CBE8')  # Soft Blue

# 3. Layout
plt.figure(figsize=(16, 10))
pos = nx.spring_layout(G, k=0.8, iterations=100, seed=42)

# 4. Draw the components
nx.draw_networkx_nodes(G, pos, 
                       node_size=3000, 
                       node_color=node_colors, 
                       edgecolors='white', 
                       linewidths=2)

# --- THE FIX FOR ARROWS ---
nx.draw_networkx_edges(G, pos, 
                       width=1.5, 
                       alpha=0.6, 
                       edge_color='#2C3E50', 
                       arrowsize=20,
                       arrowstyle='-|>',           # High-visibility arrow style
                       connectionstyle='arc3,rad=0.1', 
                       min_target_margin=25)       # Stops the arrow at the node's edge

# Draw labels
nx.draw_networkx_labels(G, pos, 
                        font_size=9, 
                        font_family='sans-serif', 
                        font_weight='bold')

plt.title("Bayesian Network: Causal Structure of Stress Predictors", 
          fontsize=18, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig("bn_diagram.png", dpi=300) 
plt.show()