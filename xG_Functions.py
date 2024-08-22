import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from statsmodels.stats.proportion import proportion_confint

## Calculates the angle of the goal in the players field of vision.
## More acute angles and larger distances will produce a smaller angle.
def angle_cal(x,y):
    import numpy as np
    angle =np.arctan(abs((7.32*x)/((x*x)+(y*y)-(3.68*3.68))))
    return angle

## Colapses the left right polarity into one foot variable.
def replace_body_part(body_part):
    if body_part in ['Left', 'Right']:
        return 'Foot'
    return body_part

## Collapses all of the non goal states into "no goal".
def replace_outcome(outcome):
    if outcome in ['Missed', 'Saved','Blocked','GoalFrame']:
        return 'No Goal'
    return outcome


# Function to plot the confusion matrix
def plot_confusion_matrix(cm, title='Confusion Matrix', labels=['No Goal', 'Goal']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Replaces goal with 1 and no goal with 0 
def binary_outcome(outcome):
    if outcome == 'No Goal':
        return 0
    elif outcome == 'Goal':
        return 1
    return outcome

## Plots the pitch given xG data.
def plot_xg_pitch(data, position_x_col, position_y_col, xg_col):
    # Conversion factor
    yard_to_meter = 0.9144

    # Convert pitch dimensions
    pitch_length = 104 * yard_to_meter
    pitch_width = 76 * yard_to_meter
    goal_area_x = 86.32 * yard_to_meter
    goal_area_y_top = 60 * yard_to_meter
    goal_area_y_bottom = 16 * yard_to_meter
    penalty_area_x = 97.97 * yard_to_meter
    penalty_area_y_top = 48 * yard_to_meter
    penalty_area_y_bottom = 27.968 * yard_to_meter
    penalty_spot_x = 92.04 * yard_to_meter
    penalty_spot_y = 38 * yard_to_meter
    center_spot_x = pitch_length / 2
    center_spot_y = pitch_width / 2
    center_circle_radius = 10 * yard_to_meter

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')

    # Plot pitch
    plt.plot([0, pitch_length], [0, 0], color="black")
    plt.plot([0, 0], [pitch_width, 0], color="black")
    plt.plot([0, pitch_length], [pitch_width, pitch_width], color="black")
    plt.plot([pitch_length, pitch_length], [pitch_width, 0], color="black")
    plt.plot([pitch_length / 2, pitch_length / 2], [0, pitch_width], color="black")

    # Goal area
    plt.plot([pitch_length, goal_area_x], [goal_area_y_top, goal_area_y_top], color="black")
    plt.plot([goal_area_x, pitch_length], [goal_area_y_bottom, goal_area_y_bottom], color="black")
    plt.plot([goal_area_x, goal_area_x], [goal_area_y_top, goal_area_y_bottom], color="black")
    plt.plot([pitch_length, penalty_area_x], [penalty_area_y_top, penalty_area_y_top], color="black")
    plt.plot([penalty_area_x, pitch_length], [penalty_area_y_bottom, penalty_area_y_bottom], color="black")
    plt.plot([penalty_area_x, penalty_area_x], [penalty_area_y_top, penalty_area_y_bottom], color="black")

    # Goal
    plt.plot([pitch_length, pitch_length], [34 * yard_to_meter, 42 * yard_to_meter], color="red")

    # Penalty spots and center circle
    penalty_spot = Circle((penalty_spot_x, penalty_spot_y), 0.25 * yard_to_meter, color="black")
    centre_spot = Circle((center_spot_x, center_spot_y), 0.5 * yard_to_meter, color="black")
    centre_circle = Circle((center_spot_x, center_spot_y), center_circle_radius, color="black", fill=False)

    # Arc for penalty area
    D = Arc((penalty_spot_x, penalty_spot_y), height=20 * yard_to_meter, width=20 * yard_to_meter, angle=0, theta1=125, theta2=235, color="black")

    ax.add_patch(centre_spot)
    ax.add_patch(centre_circle)
    ax.add_patch(penalty_spot)
    ax.add_patch(D)
    
    data['position_x_t'] = 104*yard_to_meter - data['position_x']
    
    data['position_y_t'] = data['position_y'] +34
    # Plot scatter points
    scatter = plt.scatter(data['position_x_t'], data['position_y_t'], c=data[xg_col], alpha=0.7, cmap='viridis')
    plt.colorbar(scatter, ax=ax)
    plt.show()

# uses the xG model to predict the xG given a shots data frame, with additional callibration on top.
def predict(shots, model_general):
    # Iterate through each shot in the dataframe
    for index, shot in shots.iterrows():
        # Extract the features and ensure they are converted to a float type
        features = [shot[[
            'Number_Intervening_Opponents', 'Number_Intervening_Teammates',
            'Absolute_Angle_degrees', 'distance', 'Interference_on_Shooter_encoded','BodyPart_Foot', 'BodyPart_Head', 'BodyPart_Other'
        ]].astype(float).values]
        
        if shot.iloc[0] == 0:
            shots.at[index, "XG"] = 1.2* model_foot.predict_proba(features)[:, 1][0]
        if shot.iloc[0] == 2:
            shots.at[index, "XG"] = 0.8* model_foot.predict_proba(features)[:, 1][0]
        if shot.iloc[0] == 3:
            shots.at[index, "XG"] = 1.1 *model_foot.predict_proba(features)[:, 1][0]
        
    return shots



# uses the xG model to predict the xG given a freekick data frame.
def predict_freekick(freekicks, model_fk):
    
    for index, freekick in freekicks.iterrows():
        # Extract the features and ensure they are converted to a float type
        features = [freekick[[
            'position_x', 'position_y', 'Number_Intervening_Opponents',
       'Number_Intervening_Teammates', 'Absolute_Angle_degrees', 'distance'
        ]].astype(float).values]
        print(features)
        freekicks.at[index, "XG"] = model_fk.predict_proba(features)[:, 1][0]
        
    return freekicks

## Plots the calibration curve with error bars
def plot_calibration_curve_with_error_bars(y_true, y_pred_proba, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

    # Calculate the number of data points in each bin
    bin_counts, _ = np.histogram(y_pred_proba, bins=n_bins)
    
    # Calculate confidence intervals for the true probabilities
    lower_bounds = []
    upper_bounds = []
    for count, prob in zip(bin_counts, prob_true):
        lower, upper = proportion_confint(count * prob, count, method='beta')
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    # Plot the calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Logistic Regression')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', color='orange')

    # Add error bars
    plt.errorbar(prob_pred, prob_true, yerr=[np.array(prob_true) - np.array(lower_bounds), np.array(upper_bounds) - np.array(prob_true)],
                 fmt='o', ecolor='gray', capsize=5, label='Confidence Interval')

    plt.xlabel('Predicted probability')
    plt.ylabel('True probability in each bin')
    plt.title('Calibration plot with confidence intervals')
    plt.legend()
    plt.show()

