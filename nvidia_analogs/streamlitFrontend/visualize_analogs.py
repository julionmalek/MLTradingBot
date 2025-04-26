import matplotlib.pyplot as plt

def plot_similarity(analog_dates, sims):
    labels = [d.strftime('%Y-%m-%d') for d in analog_dates]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, sims)
    ax.set_title("Cosine Similarity of Top Analogs")
    ax.set_ylabel("Similarity")
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    return fig

def plot_forward_returns(df):
    # if there's no data or no horizon column, show a dummy fig
    if df.empty or 'horizon' not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No forward‐returns data", ha='center', va='center')
        ax.axis('off')
        return [fig]

    figs = []
    for h in sorted(df['horizon'].unique()):
        sub = df[df['horizon'] == h]
        labels = [d.strftime('%Y-%m-%d') for d in sub['date']]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, sub['return_pct'])
        ax.set_title(f"{h}-Day Forward Returns for Analogs")
        ax.set_ylabel("Return (%)")
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.tight_layout()
        figs.append(fig)

    return figs

def plot_feature_profiles(current_vec, analog_vecs, feature_names):
    import pandas as pd
    from pandas.plotting import parallel_coordinates

    df = pd.DataFrame([current_vec] + analog_vecs, columns=feature_names)
    df['type'] = ['current'] + [f"analog{i+1}" for i in range(len(analog_vecs))]

    fig, ax = plt.subplots(figsize=(10, 6))
    parallel_coordinates(df, 'type', ax=ax, linewidth=2)
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_title("Feature Profile: Current vs Analogs")
    plt.tight_layout()
    return fig
