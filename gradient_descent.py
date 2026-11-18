import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

#create and graph data
np.random.seed(42)

num_samples = 500

years_of_experience = np.random.randint(2, 21, size=num_samples)

slope = (200000 - 60000) / 18
intercept = 60000

salary = slope * years_of_experience + intercept + np.random.normal(0, 10_000, num_samples )

data = {'Years_of_Experience': years_of_experience, 'Salary': salary}
df = pd.DataFrame(data)


plt.figure(figsize = (10,6))
sns.scatterplot(x ='Years_of_Experience', y = 'Salary', data = df, color = 'blue')
sns.regplot(x ='Years_of_Experience', y = 'Salary', data = df, scatter = False, color = 'red')

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Salary")

plt.show()




#Split data in train and test to detect overfitting
#overfitting: when am model emorizes noisy training examples instead of actually learning them 
X = df[["Years_of_Experience"]]
y = df["Salary"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =  0.2, random_state = 42) #assigns test_size to test_size*100 


lr = LinearRegression() #Returns R^2, the closer to one the better



lr.fit(X_train, y_train)  #learn salary formula and it's parameters, and test is held back to see how well the formula works on unseen people

train_r2 = lr.score(X_train, y_train)
print("Train R^2:", train_r2) #returns R^2, closer to 1, the better (calculates how well the already-trained model fits the training data.(X_train, y_train))

test_r2 = lr.score(X_test, y_test)
print("Test R^2:", test_r2 ) #close train and test R^2 imply that there was no overfitting


y_test_pred = lr.predict(X_test)
y_train_pred = lr.predict(X_train)
#difference between predicted values and actual values
mae = mean_absolute_error(y_test, y_test_pred)
print("Mae:",mae)

train_mse = mean_squared_error(y_train, y_train_pred )
test_mse = mean_squared_error(y_test,y_test_pred)

print("Train_mse:", train_mse)
print("Test_mse:", test_mse)


test_r2 = r2_score(y_test, y_test_pred)
print("Test R^2", test_r2)

lr_coef = lr.coef_
print("Linear Coefficient: ", lr_coef[0])


lr_intercept = lr.intercept_
print("Linear intercept:", lr_intercept)


X = np.linspace(0, 20, 100)
y = lr_coef*X + intercept

plt.scatter(X, y, label=f'y = {lr_coef[0]}x + {intercept}')

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Salary")

plt.grid()
plt.show()


X = df[["Years_of_Experience"]]
y = df["Salary"]


# Salary is in the 100,000s and years run from 2 to 20, which makes gradient descent
# slow and unstable. So work with salary in thousands and standardized years.
x_tr = X_train["Years_of_Experience"].to_numpy(dtype=float)
y_tr = y_train.to_numpy(dtype=float) / 1000
mu, sd = x_tr.mean(), x_tr.std()
xs = (x_tr - mu) / sd

w, b = 0.0, 0.0
learning_rate = 0.1
n_steps = 200
losses = []

for step in range(n_steps):
    pred = w*xs + b        # TODO: w * xs + b
    error = pred - y_tr        # TODO: pred - y_tr
    loss = np.mean(error ** 2)         # TODO: mean of error squared
    losses.append(loss)
    grad_w = 2* np.mean(error *xs)     # TODO: 2 * mean(error * xs)
    grad_b = 2 * np.mean(error)
    w = w - learning_rate * grad_w           # TODO: w - learning_rate * grad_w
    b = b - learning_rate * grad_b

plt.plot(losses); plt.xlabel("step"); plt.ylabel("loss"); plt.show()

# Convert back to dollars per year of experience and dollars
w_orig = w / sd * 1000
b_orig = (b - w * mu / sd) * 1000
print("gradient descent:", w_orig, b_orig)
print("scikit-learn:    ", lr.coef_[0], lr.intercept_)



X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train2, X_val, y_train2, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

print("sizes:", len(X_train2), len(X_val), len(X_test))

model = LinearRegression().fit(X_train2, y_train2)
sets = {"train": (X_train2, y_train2), "val": (X_val, y_val), "test": (X_test, y_test)}
for name, (A, B) in sets.items():
    print(name, "R2:", model.score(A, B), "MSE:", mean_squared_error(B, model.predict(A)))




def true_salary(x):
    return slope * np.asarray(x) + intercept     # the real line you used to make the data

grid = pd.DataFrame({"Years_of_Experience": np.arange(2, 21)})
truth = true_salary(grid["Years_of_Experience"])

for d in [0, 1, 6]:
    all_preds = []
    for k in range(20):                          # 20 different random samples of 20 people
        r = np.random.default_rng(100 + k)
        sub = df.iloc[r.choice(len(df), size=20, replace=False)]
        if d == 0:
            model = DummyRegressor(strategy="mean")
        else:
            model = make_pipeline(StandardScaler(), PolynomialFeatures(d, include_bias=False), LinearRegression())
        prediction = LinearRegression().fit(sub[["Years_of_Experience"]], sub["Salary"])
        # TODO: predict on grid and append the predictions to all_preds
        
        all_preds.append()
        plt.plot(grid["Years_of_Experience"], all_preds[-1], color="gray", alpha=0.3)

    P = np.array(all_preds)                      # shape (20, 19)
    mean_pred =                               # TODO: average over the 20 models (axis=0)
    bias_sq = ...                                # TODO: mean of (mean_pred - truth) squared
    variance = ...                               # TODO: mean over x of P.var(axis=0)
    print(f"degree {d}: bias^2 = {bias_sq:.2e}, variance = {variance:.2e}")

    plt.plot(grid["Years_of_Experience"], truth, "k", label="true line")
    plt.plot(grid["Years_of_Experience"], mean_pred, "r", label="average prediction")
    plt.title(f"degree {d}"); plt.legend(); plt.show()
