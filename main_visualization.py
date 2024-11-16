from visualization_methods import load_all_pkl_files, plot_median_residue_per_player

feature_name = 'pitch_control'
# ['pitch_control', influence']

# Define the folder containing the results
results_folder = r'saved_'+str(feature_name)+'_results'

# Load all .pkl files
all_models_results = load_all_pkl_files(results_folder)

# Visualization 1: Median residue per player across all models
plot_median_residue_per_player(all_models_results, title="Median Residue per Player Across All Models")

# # Visualization 2: Median residue per player for Game 1
# plot_median_residue_per_player(
#     all_models_results,
#     title="Median Residue per Player for Game 1",
#     filter_game=1
# )

# # Visualization 3: Median residue per player for Game 2
# plot_median_residue_per_player(
#     all_models_results,
#     title="Median Residue per Player for Game 2",
#     filter_game=2
# )

# # Visualization 4: Compare specific models (e.g., 'base', 'MLv1') for Game 2
# plot_median_residue_per_player(
#     all_models_results,
#     title="Comparison of Models (base vs MIv1) for Game 2",
#     filter_game=2,
#     filter_models=['base', 'MLv1']
# )

# # Visualization 5: Compare specific models for Game 1
# plot_median_residue_per_player(
#     all_models_results,
#     title="Comparison of Models (base vs MIv1) for Game 1",
#     filter_game=1,
#     filter_models=['base', 'MLv1']
# )