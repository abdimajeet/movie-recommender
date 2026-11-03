"""5. Movie Recommendation System: collaborative filtering on MovieLens 100K.

Problem:  predict how a user would rate movies they haven't seen, and recommend the best ones.
Dataset:  MovieLens 100K (GroupLens), 100,000 ratings (1–5 stars) from 943 users on 1,682 movies.
          Uses the official ua.base / ua.test split: 10 ratings per user are held out for testing.
Model:    matrix factorisation (biased SVD-style), trained with stochastic gradient descent:
          rating ≈ global mean + user bias + movie bias + user_vector · movie_vector

Run:  python movie_recommender.py
"""
import numpy as np
import pandas as pd

from common import dataset, r, save_result

D = dataset("ml-100k") / "ml-100k"
cols = ["user", "item", "rating", "ts"]
train = pd.read_csv(D / "ua.base", sep="\t", names=cols)
test = pd.read_csv(D / "ua.test", sep="\t", names=cols)
titles = pd.read_csv(D / "u.item", sep="|", header=None, encoding="latin-1", usecols=[0, 1], names=["item", "title"]).set_index("item").title

n_users, n_items = 943, 1682
u_tr, i_tr, y_tr = train.user.to_numpy() - 1, train.item.to_numpy() - 1, train.rating.to_numpy(float)
u_te, i_te, y_te = test.user.to_numpy() - 1, test.item.to_numpy() - 1, test.rating.to_numpy(float)

# ---- EDA ----
all_r = pd.concat([train, test])
sparsity = 1 - len(all_r) / (n_users * n_items)
counts = all_r.groupby("item").size()
top10_share = counts.sort_values(ascending=False).head(int(n_items * 0.1)).sum() / len(all_r)
mean_rating = all_r.rating.mean()

rmse = lambda pred, y: float(np.sqrt(np.mean((pred - y) ** 2)))

# ---- baselines ----
mu = y_tr.mean()
base_global = rmse(np.full_like(y_te, mu), y_te)

# ---- matrix factorisation with SGD ----
K, LR, REG, EPOCHS = 20, 0.01, 0.05, 30
rng = np.random.default_rng(42)
P = rng.normal(0, 0.1, (n_users, K))
Q = rng.normal(0, 0.1, (n_items, K))
bu, bi = np.zeros(n_users), np.zeros(n_items)

def predict(u, i):
    return np.clip(mu + bu[u] + bi[i] + np.sum(P[u] * Q[i], axis=1), 1, 5)

train_rmse, test_rmse = [], []
for epoch in range(EPOCHS):
    for idx in rng.permutation(len(y_tr)):
        u, i = u_tr[idx], i_tr[idx]
        err = y_tr[idx] - (mu + bu[u] + bi[i] + P[u] @ Q[i])
        bu[u] += LR * (err - REG * bu[u])
        bi[i] += LR * (err - REG * bi[i])
        pu = P[u].copy()
        P[u] += LR * (err * Q[i] - REG * P[u])
        Q[i] += LR * (err * pu - REG * Q[i])
    train_rmse.append(rmse(predict(u_tr, i_tr), y_tr))
    test_rmse.append(rmse(predict(u_te, i_te), y_te))
    print(f"epoch {epoch + 1:2d}  train RMSE {train_rmse[-1]:.4f}  test RMSE {test_rmse[-1]:.4f}")

best_epoch = int(np.argmin(test_rmse)) + 1

# ---- example recommendations for user 1 ----
seen = set(all_r[all_r.user == 1].item - 1)
scores = mu + bu[0] + bi + Q @ P[0]
recs = [titles[i + 1] for i in np.argsort(scores)[::-1] if i not in seen and counts.get(i + 1, 0) >= 20][:5]
liked = all_r[(all_r.user == 1) & (all_r.rating == 5)].item.map(titles).head(3).tolist()

print(f"global-mean baseline {base_global:.4f}, final test RMSE {test_rmse[-1]:.4f} (best {min(test_rmse):.4f} at epoch {best_epoch})")
print("recommendations for user 1:", recs)

save_result("movie-recommender", {
    "cfg": {"model": f"Matrix factorisation, {K} latent factors", "dataset": "MovieLens 100K, 100k ratings",
            "optimizer": "SGD", "lr": str(LR), "regularisation": str(REG), "split": "official ua.base / ua.test"},
    "metric": {"label": "Test RMSE (stars)", "value": f"{test_rmse[-1]:.3f}"},
    "metrics": [{"label": "Best epoch", "value": f"{best_epoch} ({min(test_rmse):.3f})"},
                {"label": "Average-rating baseline", "value": f"{base_global:.3f}"}],
    "curve": {"xLabel": "epoch", "yLabel": "RMSE (stars)", "x": list(range(1, EPOCHS + 1)),
              "train": r(train_rmse), "val": r(test_rmse), "trainLabel": "train RMSE", "valLabel": "test RMSE"},
    "findings": [
        f"The rating matrix is {sparsity:.1%} empty: each user has rated only a tiny fraction of the movies, which is exactly what the model has to fill in.",
        f"Popularity is uneven: the top 10% of movies receive {top10_share:.0%} of all ratings. The average rating is {mean_rating:.2f} stars.",
        f"Predicting every rating as the average gives an error of {base_global:.3f} stars. The model brings it down to {test_rmse[-1]:.3f}.",
        f"Test error is lowest at epoch {best_epoch} ({min(test_rmse):.3f}) and then creeps back up while train error keeps falling: the model starts memorising instead of learning. Stopping at epoch {best_epoch} would fix it.",
        f"User 1's 5-star films include {', '.join(liked)}. The model's top picks for them: {', '.join(recs)}.",
    ],
})
