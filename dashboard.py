'''
CREATE DASHBOARD
'''

# --- 1. IMPORT MODULES ---
if True:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA
    import streamlit as st
    import matplotlib.pyplot as plt

# --- 2. CREATE DASHBOARD ---
def run_dashboard(strategy, start, end):
        """
        Interactive Streamlit dashboard for regime analysis.
        Includes dynamic sliders for PCA components, regime count, and rolling window size.
        Also adds visualizations for PCA (2D/3D) and PCA component loadings.
        """
        st.set_page_config(layout="wide")

        st.title("Regime Detection Dashboard")

        # Sidebar controls
        st.sidebar.header("Model Parameters")

        # Create toggle to decide whether to use optimized parameters or sliders
        use_optimized = st.sidebar.checkbox("Use Optimized Parameters", value=True)

        if use_optimized:
            # Use the best params from optimization
            best_params = strategy.optimal_cluster_params(n_components_slider_range=(2, 2, 1),
                                                  n_regimes_slider_range=(2, 2, 1),
                                                  rolling_window_slider_range=(30, 90, 30))

            n_components, n_regimes, rolling_window, weighting_options = best_params['n_components'], best_params['n_regimes'], best_params['rolling_window'], best_params['weighting_options']
        else:
            # Use Streamlit sliders for parameter selection
            n_components = st.sidebar.slider("Number of PCA Components", 2, 3, 1)
            n_regimes = st.sidebar.slider("Number of Regimes (HMM States)", 2, 3, 1)
            rolling_window = st.sidebar.slider("Rolling Window (Days)", 30, 90, 30)

            # Optionally, you could still keep a default weighting options if you prefer
            weighting_options = {'time': True, 'volatility': True, 'portfolio': False}  # Default if sliders are used

        # Load and rerun the clustering logic with current parameters
        averaged_features, eig_vecs, eig_vals, X_pca, combined = strategy.cluster_analysis(n_components = n_components,
                                                                                  n_regimes = n_regimes,
                                                                                  rolling_window = rolling_window,
                                                                                  return_combined = True,
                                                                                  weighting_options = weighting_options
                                                                                  )

        tab1, tab2 = st.tabs(["Model Fit Evaluation", "Model Prediction"])

        # --- TAB 1: Model Fit Evaluation ---
        with tab1:
            st.subheader("Returns and Regimes")

            # Plot average return
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=averaged_features.index,
                y=averaged_features['daily_return'],
                mode='lines',
                name='Avg Daily Return',
                line=dict(color='black')
            ))

            # Plot each individual stock return
            return_cols = [col for col in combined.columns if 'daily_return' in col]
            for col in return_cols:
                fig.add_trace(go.Scatter(
                    x=combined.index,
                    y=combined[col],
                    mode='lines',
                    name=col,
                    line=dict(width=0.7),
                    opacity=0.4
                ))

            # Add shaded regimes
            for regime in np.unique(averaged_features['regime']):
                mask = averaged_features['regime'] == regime
                regime_df = averaged_features[mask]
                fig.add_vrect(
                    x0=regime_df.index.min(),
                    x1=regime_df.index.max(),
                    fillcolor=f"rgba({regime * 50 % 255}, {regime * 100 % 255}, {regime * 150 % 255}, 0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text=f"Regime {regime}",
                    annotation_position="top left"
                )

            fig.update_layout(
                title="Returns with Regime Overlay",
                xaxis_title="Date",
                yaxis_title="Return",
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)




            # --- PCA Component Loadings Bar Graph ---
            st.subheader("PCA Component Loadings")
            try:

                print("averaged_features.shape[1]:", averaged_features.shape[1])
                print("averaged_features.columns:", averaged_features.columns)
                print("eig_vecs:", eig_vecs)
                print("eig_vals:", eig_vals)
                print("n_components:", n_components)
                explained_variance_ratio = eig_vals / np.sum(eig_vals)
                print("explained_variance_ratio", explained_variance_ratio)

                
                num_features = averaged_features[['daily_return', 'volatility_5d', 'RSI', 'MACD', 'ADX', 'boll_width']].shape[1]
                feature_names = averaged_features[['daily_return', 'volatility_5d', 'RSI', 'MACD', 'ADX', 'boll_width']].columns
                components_df = pd.DataFrame(
                    eig_vecs.T,  # Transpose to get (2, 6) shape
                    columns=averaged_features.columns[:7],  # Use the first 6 features as column names
                    index=[f"PC{i+1}" for i in range(n_components)]  # Use the first 2 PCs (PC1, PC2)
                )
                st.dataframe(components_df)


                # Compute the explained variance manually based on eigenvalues (eigenvalues = variance explained by each component)
                idx = np.argsort(eig_vals)[::-1]
                eigenvalues = eig_vals[idx[:n_components]]  # Use the sorted eigenvalues
                explained_variance = eigenvalues / np.sum(eigenvalues)  # Explained variance ratio

                # Plot explained variance as bar chart
                fig_variance = plt.figure(figsize=(8, 6))
                plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
                plt.title("Explained Variance by PCA Components")
                plt.xlabel("Principal Components")
                plt.ylabel("Explained Variance Ratio")
                plt.xticks(range(1, len(explained_variance) + 1))
                st.pyplot(fig_variance)

            except Exception as e:
                import traceback
                st.error(f"Error displaying PCA components: {e}")
                traceback.print_exc()


            # --- PCA 2D Visualization ---
            st.subheader("PCA 2D Visualization")
            pca_2d = PCA(n_components=2)
            pca_2d_data = pca_2d.fit_transform(averaged_features.drop(columns=['regime']))

            # Ensure the number of rows matches
            if len(averaged_features) != len(pca_2d_data):
                st.error("Mismatch between the number of rows in PCA data and regime data.")
                return

            # Create 2D scatter plot with regime colors
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(pca_2d_data[:, 0], pca_2d_data[:, 1], c=averaged_features['regime'], cmap="viridis", s=100, alpha=0.7)
            plt.title("2D PCA Projection with Regimes")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.colorbar(label="Regimes")
            st.pyplot(plt.gcf())

            # --- PCA 3D Visualization ---
            st.subheader("PCA 3D Visualization")
            pca_3d = PCA(n_components=3)
            pca_3d_data = pca_3d.fit_transform(averaged_features.drop(columns=['regime']))

            # Ensure the number of rows matches
            if len(averaged_features) != len(pca_3d_data):
                st.error("Mismatch between the number of rows in PCA data and regime data.")
                return

            fig_3d = plt.figure(figsize=(10, 8))
            ax = fig_3d.add_subplot(111, projection='3d')
            scatter_3d = ax.scatter(pca_3d_data[:, 0], pca_3d_data[:, 1], pca_3d_data[:, 2], c=averaged_features['regime'], cmap="viridis", s=100, alpha=0.7)
            ax.set_title("3D PCA Projection with Regimes")
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            ax.set_zlabel("PCA Component 3")
            fig_3d.colorbar(scatter_3d, ax=ax, label="Regimes")
            st.pyplot(fig_3d)

        # --- TAB 2: Model Prediction Placeholder ---
        with tab2:
            st.subheader("Model Prediction")
            st.write("Coming soon... (rolling forecasting, reinforcement learning, etc.)")





