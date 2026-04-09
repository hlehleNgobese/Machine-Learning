"""
COMPONENT A: Hybrid Reinforcement Learning and Ensemble Learning System
South African Traffic Accident Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. DATA PREPARATION AND PREPROCESSING (10 Marks)
# ============================================================

def load_and_prepare_data(filepath=None):
    """Load dataset or generate representative synthetic data."""
    if filepath:
        df = pd.read_csv(filepath)
    else:
        print(">> No file provided. Generating synthetic SA traffic accident data for demonstration.")
        np.random.seed(42)
        n = 3000
        df = pd.DataFrame({
            'Province': np.random.choice(
                ['Gauteng', 'Western Cape', 'KwaZulu-Natal', 'Eastern Cape',
                 'Limpopo', 'Mpumalanga', 'Free State', 'North West', 'Northern Cape'], n),
            'AccidentType': np.random.choice(
                ['Head-on', 'Rear-end', 'Side-impact', 'Rollover', 'Pedestrian'], n),
            'RoadCondition': np.random.choice(
                ['Dry', 'Wet', 'Icy', 'Gravel'], n, p=[0.5, 0.3, 0.1, 0.1]),
            'LightCondition': np.random.choice(
                ['Daylight', 'Night', 'Dawn/Dusk'], n, p=[0.5, 0.35, 0.15]),
            'SpeedLimit': np.random.choice([60, 80, 100, 120], n),
            'NumberOfVehicles': np.random.randint(1, 6, n),
            'DayOfWeek': np.random.choice(
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                 'Friday', 'Saturday', 'Sunday'], n),
            'Month': np.random.randint(1, 13, n),
            'DriverAge': np.random.normal(35, 12, n).astype(int).clip(18, 75),
            'AlcoholInvolved': np.random.choice([0, 1], n, p=[0.75, 0.25]),
        })
        severity_probs = []
        for _, row in df.iterrows():
            base = 0.1
            if row['AlcoholInvolved'] == 1:
                base += 0.2
            if row['RoadCondition'] in ['Wet', 'Icy']:
                base += 0.15
            if row['LightCondition'] == 'Night':
                base += 0.1
            if row['SpeedLimit'] >= 100:
                base += 0.1
            if row['AccidentType'] == 'Head-on':
                base += 0.15
            severity_probs.append(min(base, 0.95))
        df['Severity'] = [
            np.random.choice(['Minor', 'Serious', 'Fatal'],
                             p=[1 - p, p * 0.6, p * 0.4])
            for p in severity_probs
        ]
        for col in ['DriverAge', 'SpeedLimit', 'RoadCondition']:
            mask = np.random.random(n) < 0.05
            df.loc[mask, col] = np.nan

    print(f"Dataset shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


def preprocess(df):
    """Handle missing values, encode categoricals, treat outliers."""
    print("\n--- PREPROCESSING ---")

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Imputed {col} with median ({median_val})")

    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  Imputed {col} with mode ({mode_val})")

    for col in df.select_dtypes(include=[np.number]).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            df[col] = df[col].clip(lower, upper)
            print(f"  Clipped {outliers} outliers in {col}")

    label_encoders = {}
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Severity' in cat_cols:
        cat_cols.remove('Severity')

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"  Encoded {col} ({len(le.classes_)} classes)")

    severity_map = {'Minor': 0, 'Serious': 1, 'Fatal': 2}
    if df['Severity'].dtype == 'object':
        df['Severity'] = df['Severity'].map(severity_map)

    print(f"\nFinal shape: {df.shape}")
    print(f"Target distribution:\n{df['Severity'].value_counts().sort_index()}")
    return df, label_encoders


# ============================================================
# 2. ENSEMBLE LEARNING FOR ACCIDENT PREDICTION (15 Marks)
# ============================================================

def train_ensemble_models(df):
    """Train Random Forest and XGBoost, evaluate and compare."""
    print("\n--- ENSEMBLE LEARNING ---")

    X = df.drop('Severity', axis=1)
    y = df['Severity']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    }

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results[name] = {
            'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1,
            'model': model, 'y_pred': y_pred,
        }

        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-score:  {f1:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Minor', 'Serious', 'Fatal'],
                    yticklabels=['Minor', 'Serious', 'Fatal'])
        axes[idx].set_title(f'{name} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig('data/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n--- BIAS-VARIANCE TRADE-OFF DISCUSSION ---")
    print("""
    Random Forest uses bagging which primarily reduces VARIANCE.
    Each tree trains on a bootstrap sample, making the ensemble robust to overfitting.
    However, individual trees may have high bias if max_depth is limited.

    XGBoost uses boosting which primarily reduces BIAS by sequentially correcting errors.
    Each new tree focuses on residuals from previous trees. This can lead to higher
    variance (overfitting) if not regularized properly.

    Trade-off: Random Forest is more stable but may underfit complex patterns.
    XGBoost captures more complex relationships but requires careful tuning.
    """)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (name, res) in enumerate(results.items()):
        importances = res['model'].feature_importances_
        feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)
        feat_imp.plot(kind='barh', ax=axes[idx], color='steelblue')
        axes[idx].set_title(f'{name} Feature Importance')
    plt.tight_layout()
    plt.savefig('data/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results, X_test, y_test


# ============================================================
# 3. REINFORCEMENT LEARNING FOR ACCIDENT PREVENTION (15 Marks)
# ============================================================

class AccidentPreventionMDP:
    """
    States: (road_condition, accident_frequency_level) -> 12 states
    Actions: 0=No action, 1=Safety campaign, 2=Increase enforcement, 3=Infrastructure upgrade
    Rewards: Based on reduction in accident severity
    """
    def __init__(self):
        self.road_conditions = ['Dry', 'Wet', 'Icy', 'Gravel']
        self.freq_levels = ['Low', 'Medium', 'High']
        self.actions = ['No Action', 'Safety Campaign',
                        'Increase Enforcement', 'Infrastructure Upgrade']
        self.n_states = len(self.road_conditions) * len(self.freq_levels)
        self.n_actions = len(self.actions)

    def state_index(self, road_cond, freq_level):
        return (self.road_conditions.index(road_cond) *
                len(self.freq_levels) + self.freq_levels.index(freq_level))

    def state_name(self, idx):
        rc = self.road_conditions[idx // len(self.freq_levels)]
        fl = self.freq_levels[idx % len(self.freq_levels)]
        return f"{rc}-{fl}"

    def step(self, state, action):
        rc_idx = state // len(self.freq_levels)
        fl_idx = state % len(self.freq_levels)

        if action == 0:
            reward = -5 if fl_idx >= 1 else 0
        elif action == 1:
            reward = 3 if fl_idx >= 1 else 1
        elif action == 2:
            reward = 5 if fl_idx == 2 else 2
        elif action == 3:
            reward = 7 if rc_idx >= 1 else 1

        new_fl = fl_idx
        if action > 0 and np.random.random() < 0.4:
            new_fl = max(0, fl_idx - 1)

        new_rc = rc_idx
        if np.random.random() < 0.1:
            new_rc = np.random.randint(0, len(self.road_conditions))

        new_state = new_rc * len(self.freq_levels) + new_fl
        return new_state, reward


def train_q_learning(episodes=2000, alpha=0.1, gamma=0.95,
                     epsilon_start=1.0, epsilon_end=0.01):
    """Train Q-learning agent on the accident prevention MDP."""
    print("\n--- REINFORCEMENT LEARNING (Q-Learning) ---")

    mdp = AccidentPreventionMDP()
    Q = np.zeros((mdp.n_states, mdp.n_actions))
    epsilon_decay = (epsilon_start - epsilon_end) / episodes
    rewards_per_episode = []
    epsilon = epsilon_start

    for ep in range(episodes):
        state = np.random.randint(0, mdp.n_states)
        total_reward = 0
        for _ in range(50):
            if np.random.random() < epsilon:
                action = np.random.randint(0, mdp.n_actions)
            else:
                action = np.argmax(Q[state])
            next_state, reward = mdp.step(state, action)
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            total_reward += reward
        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_end, epsilon - epsilon_decay)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    smoothed = pd.Series(rewards_per_episode).rolling(50).mean()
    axes[0].plot(rewards_per_episode, alpha=0.3, color='blue')
    axes[0].plot(smoothed, color='red', linewidth=2)
    axes[0].set_title('Q-Learning Convergence')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')

    policy = np.argmax(Q, axis=1)
    policy_matrix = policy.reshape(len(mdp.road_conditions), len(mdp.freq_levels))
    sns.heatmap(policy_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1],
                xticklabels=mdp.freq_levels, yticklabels=mdp.road_conditions)
    axes[1].set_title('Learned Policy (Action Index)')
    axes[1].set_xlabel('Accident Frequency')
    axes[1].set_ylabel('Road Condition')
    plt.tight_layout()
    plt.savefig('data/rl_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nLearned Policy:")
    print(f"{'State':<20} {'Best Action':<25} {'Q-Value':<10}")
    print("-" * 55)
    for s in range(mdp.n_states):
        best_a = np.argmax(Q[s])
        print(f"{mdp.state_name(s):<20} {mdp.actions[best_a]:<25} {Q[s, best_a]:.2f}")

    return Q, mdp, rewards_per_episode


# ============================================================
# 4. SYSTEM INTEGRATION AND EVALUATION (10 Marks)
# ============================================================

def integrate_system(ensemble_results, Q, mdp, df):
    """Integrate ensemble predictions with RL intervention strategy."""
    print("\n--- SYSTEM INTEGRATION ---")

    best_model_name = max(ensemble_results, key=lambda k: ensemble_results[k]['F1'])
    print(f"Best ensemble model: {best_model_name} "
          f"(F1={ensemble_results[best_model_name]['F1']:.4f})")

    policy = np.argmax(Q, axis=1)

    print("\n--- POLICY EVALUATION TABLE ---")
    print(f"{'Road Condition':<15} {'Freq Level':<12} "
          f"{'Recommended Action':<25} {'Expected Reward':<15}")
    print("-" * 67)
    for s in range(mdp.n_states):
        best_a = policy[s]
        parts = mdp.state_name(s).split('-')
        print(f"{parts[0]:<15} {parts[1]:<12} "
              f"{mdp.actions[best_a]:<25} {Q[s, best_a]:.2f}")

    print("\n--- EXPLORATION-EXPLOITATION TRADE-OFF ---")
    print("""
    The Q-learning agent uses epsilon-greedy exploration:
    - Early training (high epsilon): explores random actions to discover reward structure.
    - Late training (low epsilon): exploits learned Q-values to maximize cumulative reward.

    Trade-off: Too much exploration wastes resources on suboptimal actions.
    Too little exploration may miss better long-term strategies.
    The decaying epsilon schedule balances this.

    ETHICAL IMPLICATIONS:
    - Automated road safety decisions affect human lives.
    - The system should serve as decision-support, not replace human judgment.
    - Bias in historical data can lead to inequitable resource allocation.
    - Transparency: stakeholders must understand why interventions are recommended.
    - Regular model retraining is needed as conditions evolve.
    """)


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_component_a(filepath=None):
    """Execute full Component A pipeline."""
    print("=" * 60)
    print("COMPONENT A: Hybrid RL + Ensemble Learning System")
    print("=" * 60)

    df = load_and_prepare_data(filepath)
    df, encoders = preprocess(df)
    df.to_csv('data/preprocessed_accidents.csv', index=False)
    print("\nPreprocessed data saved to data/preprocessed_accidents.csv")

    results, X_test, y_test = train_ensemble_models(df)
    Q, mdp, rewards = train_q_learning()
    integrate_system(results, Q, mdp, df)

    return df, results, Q, mdp


if __name__ == '__main__':
    run_component_a()
