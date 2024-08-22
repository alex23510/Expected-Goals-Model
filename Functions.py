import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
import random 
from scipy.stats import beta


# Formats the wyscout dataframe.
def format_shot_type(row):
  
    if row["Subtype"] == "HEAD-OFF TARGET-OUT" :
        row["BodyPart_Foot"] = False
        row["BodyPart_Head"] = True
        row["BodyPart_Other"]=False
            
    elif row["Subtype"] == "HEAD-ON TARGET-GOAL" :
        row["BodyPart_Foot"] = False
        row["BodyPart_Head"] = True    
        row["BodyPart_Other"]=False
    
    elif row["Subtype"] == "OFF TARGET-HEAD-OUT" :
        row["BodyPart_Foot"] = False
        row["BodyPart_Head"] = True    
        row["BodyPart_Other"]=False
                
    elif row["Subtype"] == "ON TARGET-SAVED" :
        row["BodyPart_Foot"] = True
        row["BodyPart_Head"] = False
        row["BodyPart_Other"]=False
            
    elif row["Subtype"] == "ON TARGET-GOAL" :
        row["BodyPart_Foot"] = True
        row["BodyPart_Head"] = False
        row["BodyPart_Other"]=False
    
    elif row["Subtype"] == "OFF TARGET-OUT" :
        row["BodyPart_Foot"] = True
        row["BodyPart_Head"] = False
        row["BodyPart_Other"]=False
            
    elif row["Subtype"] == "BLOCKED":
        row["BodyPart_Foot"] = True
        row["BodyPart_Head"] = False
        row["BodyPart_Other"]=False
    return row

## This is a function to sanity check the transformations - not used in final implementation 
def summarize_player_positions_by_period(df):
    period1 = df[df["Period"]==1]
    period2 = df[df["Period"]==2]
    
    mean_x_pl2_p1 = period1["Player2 X"].mean()
    mean_y_pl2_p1 = period1["Player2 Y"].mean()
    mean_x_pl3_p1 = period1["Player3 X"].mean()
    mean_y_pl3_p1 = period1["Player3 Y"].mean()
    
    mean_x_pl2_p2 = period2["Player2 X"].mean()
    mean_y_pl2_p2 = period2["Player2 Y"].mean()
    mean_x_pl3_p2 = period2["Player3 X"].mean()
    mean_y_pl3_p2 = period2["Player3 Y"].mean()
    
    
    print(f"Period 1 - Player 2: Mean X: {mean_x_pl2_p1:.2f}, Mean Y: {mean_y_pl2_p1:.2f}")
    print(f"Period 2 - Player 2: Mean X: {mean_x_pl2_p2:.2f}, Mean Y: {mean_y_pl2_p2:.2f}")
    print(f"Period 1 - Player 3: Mean X: {mean_x_pl3_p1:.2f}, Mean Y: {mean_y_pl3_p1:.2f}")
    print(f"Period 2 - Player 3: Mean X: {mean_x_pl3_p2:.2f}, Mean Y: {mean_y_pl3_p2:.2f}")

    return

# Transforms the data frame, this function is not generalisable. 
def transform_data_frame(df, home_away):
    # Function to transform the coordinates so teams play on the same half of the pitch each half 
    
    # Define the new column headings based on home_away parameter
    if home_away == "Home":
        new_column_headings = [
            "Period", "Frame", "Time [s]", 
            "Player1 X", "Player1 Y", 
            "Player2 X", "Player2 Y", 
            "Player3 X", "Player3 Y", 
            "Player4 X", "Player4 Y",
            "Player5 X", "Player5 Y",
            "Player6 X", "Player6 Y",
            "Player7 X", "Player7 Y",
            "Player8 X", "Player8 Y",
            "Player9 X", "Player9 Y",
            "Player10 X", "Player10 Y", 
            "Player11 X", "Player11 Y",
            "Player12 X", "Player12 Y", 
            "Player13 X", "Player13 Y", 
            "Player14 X", "Player14 Y", 
            "Ball X", "Ball Y"
        ]
    elif home_away == "Away":
        new_column_headings = [
            "Period", "Frame", "Time [s]", 
            "Player25 X", "Player25 Y", 
            "Player15 X", "Player15 Y", 
            "Player16 X", "Player16 Y", 
            "Player17 X", "Player17 Y", 
            "Player18 X", "Player18 Y",
            "Player19 X", "Player19 Y",
            "Player20 X", "Player20 Y",
            "Player21 X", "Player21 Y",
            "Player22 X", "Player22 Y",
            "Player23 X", "Player23 Y",
            "Player24 X", "Player24 Y", 
            "Player26 X", "Player26 Y", 
            "Ball X", "Ball Y"
        ]
        

    # Ensure the length of new_column_headings matches the number of columns in the DataFrame
    assert len(new_column_headings) == len(df.columns), "New column headings list must match the number of columns in the DataFrame"

    # Assign the new column headings
    df.columns = new_column_headings

    # Drop the first row and reset the index
    df = df[2:]
    df.reset_index(drop=True, inplace=True)


    return df

# Transforms the positions in the correct coordinate reference frame. 
def transform_positions(df):
    # Ensure all numeric columns are correctly formatted
    df = df.copy()

    # Identify columns that end with 'X' and 'Y'
    x_columns = [col for col in df.columns if col.endswith('X')]
    y_columns = [col for col in df.columns if col.endswith('Y')]

    # Function to split and convert concatenated string values
    def split_and_convert(value):
        try:
            if isinstance(value, str):
                # Split concatenated values and convert to float, then return the first value
                return float(value.split()[0])
            else:
                return float(value)
        except (ValueError, TypeError):
            return value
    
    # Apply the function to the relevant columns
    for col in x_columns + y_columns:
        df[col] = df[col].apply(split_and_convert)
    
    # Create a mask for the specified period
    period_mask = df['Period'] == 2

    # Apply the transformation (1 - X) for X positions and (1 - Y) for Y positions in the specified period
    df.loc[period_mask, x_columns] = 1 - df.loc[period_mask, x_columns]
    df.loc[period_mask, y_columns] = 1 - df.loc[period_mask, y_columns]
    
    # Multiply all X positions by 105
    df[x_columns] = df[x_columns] * 105
    # Multiply all Y positions by 68
    df[y_columns] = df[y_columns] * 68

    return df

## Detect players for the intervening players, and inference on shots. 
def detect_players(home, away, shots):
    shots["Number_Intervening_Teammates"]=0
    shots["Number_Intervening_Opponents"]=0
    shots["Interference_on_Shooter"] = 0
    
    goal_post_left_home = (105, 34 - 3.66)
    goal_post_right_home = (105, 34 + 3.66)
    goal_post_left_away = (0, 34 - 3.66)
    goal_post_right_away = (0, 34 + 3.66)
    
    
    # Columns ending with 'X' and 'Y' for player positions
    x_columns_home = [col for col in home.columns if col.endswith('X')]
    y_columns_home = [col for col in home.columns if col.endswith('Y')]
    x_columns_away = [col for col in away.columns if col.endswith('X')]
    y_columns_away = [col for col in away.columns if col.endswith('Y')]
    
    
    # Ensure the Frame columns are of the same data type
    home["Frame"] = home["Frame"].astype(str)
    away["Frame"] = away["Frame"].astype(str)
    shots["Start Frame"] = shots["Start Frame"].astype(str)
    
    for i in range(len(shots)):
        # Get the frame and create a mask 
        frame = shots["Start Frame"].iloc[i]
        shooter_pos = (shots["Start X"].iloc[i], shots["Start Y"].iloc[i])
        # Create masks
        frame_mask_away = away["Frame"] == frame
        frame_mask_home = home["Frame"] == frame 
        
        # Check if masks select the correct rows
        home_selected = home.loc[frame_mask_home]
        away_selected = away.loc[frame_mask_away]
        
        # Extract player positions, handling missing columns
        home_players_x = home_selected[x_columns_home].dropna(axis=1, how='all')
        home_players_y = home_selected[y_columns_home].dropna(axis=1, how='all')
        away_players_x = away_selected[x_columns_away].dropna(axis=1, how='all')
        away_players_y = away_selected[y_columns_away].dropna(axis=1, how='all')
        
        # Combine X and Y positions into a single DataFrame for home and away players
        home_players = pd.DataFrame()
        away_players = pd.DataFrame()
        
        for col_x, col_y in zip(home_players_x.columns, home_players_y.columns):
            player_data = pd.DataFrame({
                'Player': [col_x.split()[0]],  # Assuming column name format 'Player1 X'
                'x': home_players_x[col_x].values,
                'y': home_players_y[col_y].values,
                'Team': 'Home',
                'Frame': frame
            })
            home_players = pd.concat([home_players, player_data], ignore_index=True)
        
        for col_x, col_y in zip(away_players_x.columns, away_players_y.columns):
            player_data = pd.DataFrame({
                'Player': [col_x.split()[0]],  # Assuming column name format 'Player1 X'
                'x': away_players_x[col_x].values,
                'y': away_players_y[col_y].values,
                'Team': 'Away',
                'Frame': frame
            })
            away_players = pd.concat([away_players, player_data], ignore_index=True)
        
        # Remove rows where Player is 'Ball'
        home_players = home_players[home_players['Player'] != 'Ball']
        away_players = away_players[away_players['Player'] != 'Ball']
        
        # Combine home and away players into a single DataFrame
        combined_frame = pd.concat([home_players, away_players], ignore_index=True)
        
        
        team = shots["Team"].iloc[i]
        
        
        if team == "Home":
            df_with_cone = check_players_in_shot_cone(combined_frame, shooter_pos, goal_post_left_away, goal_post_right_away)
            plot_field_with_shot_cone(df_with_cone, shooter_pos, goal_post_left_away, goal_post_right_away)
            interference = interference_with_shooter(away_players, shooter_pos)
    
        if team == "Away":
            df_with_cone = check_players_in_shot_cone(combined_frame, shooter_pos, goal_post_left_home, goal_post_right_home)
            plot_field_with_shot_cone(df_with_cone, shooter_pos, goal_post_left_home, goal_post_right_home)
            interference = interference_with_shooter(home_players, shooter_pos)
            
        # Count the number of True values in 'in_shot_cone' for each team
        true_counts = df_with_cone.groupby('Team')['in_shot_cone'].sum()

        # Assign counts to individual variables
        home_true_count = true_counts.get('Home', 0)  # Default to 0 if the team is not present
        away_true_count = true_counts.get('Away', 0)  # Default to 0 if the team is not present
        
        
        if team == "Home":
            shots["Number_Intervening_Teammates"].iloc[i] = home_true_count
            shots["Number_Intervening_Opponents"].iloc[i] = away_true_count
        elif team == "Away":
            shots["Number_Intervening_Teammates"].iloc[i] = away_true_count
            shots["Number_Intervening_Opponents"].iloc[i] = home_true_count
        
        if interference== 0 :
            shots["Interference_on_Shooter"].iloc[i] = "Low"
        elif interference ==1:
            shots["Interference_on_Shooter"].iloc[i] = "Medium"
        elif interference > 1:
            shots["Interference_on_Shooter"].iloc[i] = "High"

    return shots

## calculates the inference on the shooter, any players within 2m are interfering
## personally i think this is too far away but i was trying to mimick the spread in the training data set.
def interference_with_shooter(players, shooter_pos):
    interference = 0
    shooter_x, shooter_y = shooter_pos  # Unpack shooter position into x and y coordinates
    
    for idx, player in players.iterrows():
        x_dist = player["x"] - shooter_x
        y_dist = player["y"] - shooter_y
        
        distance = (x_dist**2 + y_dist**2)**0.5
        # Add logic to check interference
        if distance < 2:  #
            interference += 1
    
    return interference

## calculates the triangle area between the verticies left post, right post and the ball. 
def point_in_shot_cone(shooter_pos, player_pos, goal_post_left, goal_post_right):
    def triangle_area(x1, y1, x2, y2, x3, y3):
        return abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0
    
    x_s, y_s = shooter_pos
    x_p, y_p = player_pos
    x_l, y_l = goal_post_left
    x_r, y_r = goal_post_right
    
    area_full = triangle_area(x_s, y_s, x_l, y_l, x_r, y_r)
    
    area1 = triangle_area(x_p, y_p, x_s, y_s, x_l, y_l)
    area2 = triangle_area(x_p, y_p, x_s, y_s, x_r, y_r)
    area3 = triangle_area(x_p, y_p, x_l, y_l, x_r, y_r)
    
    return abs(area1 + area2 + area3 - area_full) < 1e-6

## finds the players in the above cone. 
def check_players_in_shot_cone(df, shooter_pos, goal_post_left, goal_post_right):
    df['in_shot_cone'] = df.apply(lambda row: point_in_shot_cone(shooter_pos, (row['x'], row['y']), goal_post_left, goal_post_right), axis=1)
    return df

## plots the result. 
def plot_field_with_shot_cone(df, shooter_pos, goal_post_left, goal_post_right):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Pitch dimensions
    pitch_length = 105
    pitch_width = 68
    
    # Plot pitch boundaries
    plt.plot([0, 0, pitch_length, pitch_length, 0], [0, pitch_width, pitch_width, 0, 0], color="green")
    
    # Plot goals
    plt.plot([0, 0], [34-3.66, 34+3.66], color="red", linewidth=5)
    plt.plot([pitch_length, pitch_length], [34-3.66, 34+3.66], color="red", linewidth=5)
    
    # Plot shooter
    plt.scatter(shooter_pos[0], shooter_pos[1], color='blue', s=100, label='Shooter')
    
    # Plot players
    in_cone = df[df['in_shot_cone'] == True]
    out_cone = df[df['in_shot_cone'] == False]
    
    plt.scatter(in_cone['x'], in_cone['y'], color='yellow', s=100, label='Player in Shot Cone')
    plt.scatter(out_cone['x'], out_cone['y'], color='black', s=100, label='Player out of Shot Cone')
    
    # Plot shot cone
    plt.plot([shooter_pos[0], goal_post_left[0]], [shooter_pos[1], goal_post_left[1]], color="blue", linestyle="--")
    plt.plot([shooter_pos[0], goal_post_right[0]], [shooter_pos[1], goal_post_right[1]], color="blue", linestyle="--")
    plt.plot([goal_post_left[0], goal_post_right[0]], [goal_post_left[1], goal_post_right[1]], color="blue", linestyle="--")
    
    # Setting axis limits
    plt.xlim(-5, pitch_length+5)
    plt.ylim(-5, pitch_width+5)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title('Shot Cone Visualization')
    plt.xlabel('Pitch Length (meters)')
    plt.ylabel('Pitch Width (meters)')
    
    plt.show()

# transforms the x y wyscout coordinates into the correct coordinate reference system for the ml model    
def ML_X_Y(df):
    # Initialize new columns with original values
    df['ML X'] = df['Start X']
    df['ML Y'] = df['Start Y']
    
    # Apply transformations based on the team
    df.loc[df['Team'] == 'Home', 'ML Y'] = 34 - df.loc[df['Team'] == 'Home', 'Start Y']
    df.loc[df['Team'] == 'Away', 'ML X'] = 105 - df.loc[df['Team'] == 'Away', 'Start X']
    df.loc[df['Team'] == 'Away', 'ML Y'] = 34 - df.loc[df['Team'] == 'Away', 'Start Y']
    
    return df

## calculates the angle of goal in the shooters field of view. 
def angle_cal(x,y):
    angle =np.arctan(abs((7.32*x)/((x*x)+(y*y)-(3.68*3.68))))
    return angle


## plots the goals and xG over time 
def plot_xg_and_goals_over_time(df):
    """
    Plot Expected Goals (XG) and goals scored over time for each team.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the following columns:
                           - 'Team': The team name
                           - 'Subtype': The subtype of the shot event
                           - 'Start Time [s]': The start time of the shot in seconds
                           - 'XG': The expected goals value of the shot
    
    Returns:
    None
    """
    
    # Convert Start Time [s] to minutes
    df['Time [min]'] = df['Start Time [s]'] / 60

    # Identify goals
    df['Goal'] = df['Subtype'].str.contains('GOAL').astype(int)

    # Group by Team and Time, and sum the XG and Goal columns
    xg_data = df.groupby(['Team', 'Time [min]']).agg({'XG': 'sum', 'Goal': 'sum'}).reset_index()

    # Define colors for each team
    team_colors = {
        'Home': 'orange',
        'Away': 'blue'
    }

    # Plotting
    plt.figure(figsize=(14, 8))

    for team in xg_data['Team'].unique():
        team_data = xg_data[xg_data['Team'] == team]
        cumulative_xg = team_data['XG'].cumsum()
        cumulative_goals = team_data['Goal'].cumsum()
        color = team_colors.get(team, 'black')  # Default to black if team color is not defined

        plt.plot(team_data['Time [min]'], cumulative_xg, label=f'{team} XG', marker='o', markersize=8, linestyle='--', color=color)
        plt.plot(team_data['Time [min]'], cumulative_goals, label=f'{team} Goals', marker='x', markersize=8, linestyle='-', color=color)
        
        goals = df[(df['Team'] == team) & (df['Goal'] == 1)]
        
        # Add annotations for goals and use different colors for each team
        for i, (x, y) in enumerate(zip(goals['Time [min]'], goals['Goal'].cumsum())):
            plt.annotate(f'Goal {i+1}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color=color)

        # Shade region between XG and Goals
        plt.fill_between(team_data['Time [min]'], cumulative_xg, cumulative_goals, 
                         where=(cumulative_goals >= cumulative_xg), facecolor='green', alpha=0.3, interpolate=True)
        plt.fill_between(team_data['Time [min]'], cumulative_xg, cumulative_goals, 
                         where=(cumulative_goals < cumulative_xg), facecolor='red', alpha=0.3, interpolate=True)

    plt.xlabel('Time (minutes)', fontsize=14)
    plt.ylabel('Cumulative XG / Goals', fontsize=14)
    plt.title('XG and Goals Scored Over Time', fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


## calcualtes a binomial table, with a list of xG values. 
def calculate_individual_binomial_probabilities(xg_values):
    """
    Calculate the individual binomial probabilities of scoring 0 to n goals given a list of XG values.

    Parameters:
    xg_values (list): List of XG values for each shot.

    Returns:
    pd.DataFrame: DataFrame containing individual probabilities for scoring 0 to n goals.
    """
    n_shots = len(xg_values)
    max_goals = n_shots
    individual_probabilities = np.zeros((n_shots + 1, max_goals + 1))

    # Initial probabilities for 0 shots (i.e., 100% chance of 0 goals)
    individual_probabilities[0, 0] = 1.0

    for i in range(1, n_shots + 1):
        xg = xg_values[i - 1]
        for k in range(i + 1):
            if k == 0:
                # Probability of scoring 0 goals up to shot i
                individual_probabilities[i, k] = individual_probabilities[i - 1, k] * (1 - xg)
            else:
                # Probability of scoring k goals up to shot i
                individual_probabilities[i, k] = (individual_probabilities[i - 1, k - 1] * xg +
                                                  individual_probabilities[i - 1, k] * (1 - xg))

    # Convert to DataFrame, excluding the first row which is for 0 shots
    return pd.DataFrame(individual_probabilities[1:], columns=[f'Goals {k}' for k in range(max_goals + 1)])

## plots the binomial probibilites, this function is fairly redundant, needs updating in the long term. 
def plot_stacked_binomial_probabilities(df, show_plot):
    """
    Plot stacked binomial probabilities of goals scored over time for each team and return the individual probability DataFrames.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the following columns:
                           - 'Team': The team name
                           - 'Subtype': The subtype of the shot event
                           - 'Start Time [s]': The start time of the shot in seconds
                           - 'XG': The expected goals value of the shot
    
    Returns:
    dict: Dictionary containing individual probability DataFrames for each team.
    """
    
    # Convert Start Time [s] to minutes
    df['Time [min]'] = df['Start Time [s]'] / 60

    # Identify goals
    df['Goal'] = df['Subtype'].str.contains('GOAL').astype(int)

    # Group by Team and Time, and sum the XG and Goal columns
    xg_data = df.groupby(['Team', 'Time [min]']).agg({'XG': 'sum', 'Goal': 'sum'}).reset_index()

    # Define colors for each team
    team_colors = {
        'Home': 'Oranges',
        'Away': 'Blues'
    }
    
    prob_dfs = {}

    for team in xg_data['Team'].unique():
        team_data = xg_data[xg_data['Team'] == team]
        xg_values = team_data['XG'].tolist()
        time_values = team_data['Time [min]'].tolist()
        
        # Calculate individual binomial probabilities
        individual_probabilities_df = calculate_individual_binomial_probabilities(xg_values)
        individual_probabilities_df['Time [min]'] = time_values
        prob_dfs[team] = individual_probabilities_df
        
        # Plotting
        plt.figure(figsize=(14, 8))
        base_color = team_colors[team]
        shades = [plt.cm.get_cmap(base_color)(i / (len(individual_probabilities_df.columns)-1)) for i in range(len(individual_probabilities_df.columns))]
        for i, col in enumerate(individual_probabilities_df.columns[:-1]):
            plt.fill_between(time_values, 0 if i == 0 else individual_probabilities_df.iloc[:, i - 1],
                             individual_probabilities_df.iloc[:, i], color=shades[i], label=f'{col}')

        # Add cumulative goal lines
        cumulative_goals = team_data['Goal'].cumsum()
        plt.plot(time_values, cumulative_goals, marker='x', markersize=8, linestyle='-', color='black', label=f'{team} Goals')

        # Add annotations for goals
        goals = df[(df['Team'] == team) & (df['Goal'] == 1)]
        for i, (x, y) in enumerate(zip(goals['Time [min]'], goals['Goal'].cumsum())):
            plt.annotate(f'Goal {i+1}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='black')

        plt.xlabel('Time (minutes)', fontsize=14)
        plt.ylabel('Probability', fontsize=14)
        plt.title(f'Stacked Binomial Probabilities of Goals Over Time - {team}', fontsize=16)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        if show_plot == True:
            plt.show()
        if show_plot == False:
            plt.close()
        
    return prob_dfs

# combines two binomial data frames 
def combine_prob_dataframes(df1, df2):
    """
    Combine two probability DataFrames with different numbers of columns, 
    adding a source column to indicate the origin of each row.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.

    Returns:
    pd.DataFrame: The combined DataFrame.
    """
    # Add a column to indicate which DataFrame the row came from
    df1['Team'] = 'Home'
    df2['Team'] = 'Away'

    # Determine the maximum number of goals columns
    max_goals = max(df1.shape[1] - 2, df2.shape[1] - 2)

    # Ensure both DataFrames have the same columns
    for df in [df1, df2]:
        for i in range(max_goals + 1):
            col_name = f'Goals {i}'
            if col_name not in df.columns:
                df[col_name] = 0
            

    # Reorder the columns to align them
    df1 = df1[['Team'] + [f'Goals {i}' for i in range(max_goals + 1)] + ['Time [min]']]
    df2 = df2[['Team'] + [f'Goals {i}' for i in range(max_goals + 1)] + ['Time [min]']]

    # Add the initial row with 100% probability of no goals
    initial_row = pd.DataFrame({
        'Team': ['Home'],
        **{f'Goals {i}': [1.0] if i == 0 else [0.0] for i in range(max_goals + 1)},
        'Time [min]': [0]
    })
    df1 = pd.concat([initial_row, df1], ignore_index=True)

    initial_row = pd.DataFrame({
        'Team': ['Away'],
        **{f'Goals {i}': [1.0] if i == 0 else [0.0] for i in range(max_goals + 1)},
        'Time [min]': [0]
    })
    df2 = pd.concat([initial_row, df2], ignore_index=True)

    # Combine the DataFrames
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    return combined_df.sort_values("Time [min]").reset_index(drop=True)


## Caluculates the rolling win chance based off of the binomail dataframe
def calculate_rolling_win_chance(df):
    results = []
    home_row = None
    away_row = None
    time = None

    for index, row in df.iterrows():
        if row["Team"] == "Home":
            home_row = row
            time = row["Time [min]"]
        elif row["Team"] == "Away":
            away_row = row
            time = row["Time [min]"]

        if home_row is not None and away_row is not None:
            home_probabilities = home_row.filter(like='Goals').values
            away_probabilities = away_row.filter(like='Goals').values

            # Create a matrix of combined probabilities
            combined_matrix = np.outer(home_probabilities, away_probabilities)

            # Sum the probabilities according to the regions
            away_win_prob = np.triu(combined_matrix, 1).sum()  # Above the diagonal
            home_win_prob = np.tril(combined_matrix, -1).sum()  # Below the diagonal
            draw_prob = np.diag(combined_matrix).sum()  # Diagonal

            # Append the results
            results.append({
                'Time [min]': time,
                'Home Win Probability': home_win_prob,
                'Away Win Probability': away_win_prob,
                'Draw Probability': draw_prob
            })

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

## Plotting function 
def plot_win_probabilities(data):
    """
    Plots the Home Win, Away Win, and Draw Probabilities over time.

    Parameters:
    data (dict): A dictionary containing 'Time [min]', 'Home Win Probability', 'Away Win Probability', and 'Draw Probability' lists.
    """
    df = pd.DataFrame(data)

    # Plotting the data
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time [min]'], df['Home Win Probability'], label='Home Win Probability', marker='o')
    plt.plot(df['Time [min]'], df['Away Win Probability'], label='Away Win Probability', marker='o')
    plt.plot(df['Time [min]'], df['Draw Probability'], label='Draw Probability', marker='o')

    plt.xlabel('Time [min]')
    plt.ylabel('Probability')
    plt.title('Home Win, Away Win, and Draw Probabilities Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

## This function is used in the stratagey function to generate a list of a 1000 shot samples, to simulate the variability in football, based off the average xG per shot. The distribution is not perfect, but its fine for a proof of concept. 
def generate_shot_samples(desired_mean, min_val =0.01, max_val=1, n_samples=1000):
    # Calculate the corresponding mean for the Beta distribution
    beta_mean = (desired_mean - min_val) / (max_val - min_val)
    
    # Choose k to determine the shape parameters
    k = 2  # You can adjust this value for different variances
    alpha = beta_mean * k
    beta_param = (1 - beta_mean) * k
    
    # Generate random values from the Beta distribution
    samples = beta.rvs(alpha, beta_param, size=n_samples)
    
    # Scale and shift the samples to fit within the desired range
    scaled_samples = samples * (max_val - min_val) + min_val
    
    return scaled_samples

## Predicts the win chance by taking different statrageys.
def predict_win_chance_with_stratagey(minute, team, strategy, shots_df):
    
    strategies = {
    "attacking": {
        "no_shots_for": 1.4,
        "no_shots_against": 1.4,
        "xg_per_shot_for": 1.4,
        "xg_per_shot_against": 1.4
    },
    "neutral": {
        "no_shots_for": 1.0,
        "no_shots_against": 1.0,
        "xg_per_shot_for": 1.0,
        "xg_per_shot_against": 1.0
    },
    "defensive": {
        "no_shots_for": 0.7,
        "no_shots_against": 0.7,
        "xg_per_shot_for": 0.7,
        "xg_per_shot_against": 0.7
    }
}


    pre_prediction_df = shots_df[['Team', 'XG', 'Time [min]', 'Goal']]
    pre_prediction_df = pre_prediction_df[pre_prediction_df["Time [min]"]<minute]
        
    # Initialize counters for goals
    home_goals = 0
    away_goals = 0
    home_shots_no = 0
    home_shots_quality = 0
    away_shots_no = 0
    away_shots_quality = 0

    for index, row in pre_prediction_df.iterrows():
    
        if row['Team'] == 'Home':
            home_shots_no +=1
            home_shots_quality += row["XG"]
            if row['Goal'] == 1:
                home_goals += 1
        
        elif row['Team'] == 'Away':
            away_shots_no+= 1        
            away_shots_quality += row["XG"]
            if row['Goal'] == 1:
                away_goals += 1
        
    avg_home_shot_quality = home_shots_quality/ home_shots_no
    avg_away_shot_quality = away_shots_quality/ away_shots_no
    avg_home_shots_per_min =  home_shots_no/minute 
    avg_away_shots_per_min =  away_shots_no/minute 
    
    if team == "Home":
        new_avg_home_shot_quality = avg_home_shot_quality* strategies[strategy]["xg_per_shot_for"]
        new_avg_home_shots_per_min = avg_home_shots_per_min * strategies[strategy]["no_shots_for"]
        new_avg_away_shot_quality = avg_away_shot_quality* strategies[strategy]["xg_per_shot_against"]
        new_avg_away_shots_per_min = avg_away_shots_per_min * strategies[strategy]["no_shots_against"]
        
    if team == "Away":
        new_avg_away_shot_quality = avg_away_shot_quality* strategies[strategy]["xg_per_shot_for"]
        new_avg_away_shots_per_min = avg_away_shots_per_min * strategies[strategy]["no_shots_for"]
        new_avg_home_shot_quality = avg_home_shot_quality* strategies[strategy]["xg_per_shot_against"]
        new_avg_home_shots_per_min = avg_home_shots_per_min * strategies[strategy]["no_shots_against"]
        
    no_simulations = 1000
    home_shots_list = generate_shot_samples(new_avg_away_shot_quality)
    away_shots_list = generate_shot_samples(new_avg_away_shot_quality)
    
    home_wins =0
    draws =0
    away_wins =0 
    avg_points_home=0 
    avg_points_away=0
    avg_new_home_goals =0
    avg_new_away_goals =0
    
    for i in range(no_simulations):
        
        end_time = 90 + random.randint(2, 8)
        new_home_goals=0
        new_away_goals=0
        
        for j in range(minute, end_time):
            home_chance = np.random.choice([1, 0], p=[new_avg_home_shots_per_min, 1-new_avg_home_shots_per_min])
            away_chance = np.random.choice([1, 0], p=[new_avg_away_shots_per_min, 1-new_avg_away_shots_per_min])
            if home_chance == 1:
                shot = home_shots_list[np.random.randint(0, 1000)]
                new_home_goals +=  np.random.choice([1, 0], p=[shot, 1-shot])
            if away_chance == 1:
                shot = away_shots_list[np.random.randint(0, 1000)]
                new_away_goals +=  np.random.choice([1, 0], p=[shot, 1-shot])
        if (new_home_goals+home_goals)> ((new_away_goals+away_goals)):
            home_wins += 1
            avg_points_home+= 3 
            avg_points_away+=0
            
        elif (new_home_goals+home_goals)== ((new_away_goals+away_goals)):
            avg_points_home+=1
            avg_points_away+=1
            draws +=1
        elif (new_home_goals+home_goals)< ((new_away_goals+away_goals)):
            avg_points_home+= 0 
            avg_points_away+=3
            away_wins +=1 
    
        avg_new_home_goals += new_home_goals
        avg_new_away_goals += new_away_goals
    
    avg_new_home_goals = avg_new_home_goals/no_simulations
    avg_new_away_goals = avg_new_away_goals/no_simulations
    avg_points_home= avg_points_home/no_simulations 
    avg_points_away= avg_points_away/no_simulations
    #print("Home goals: ", home_goals,"sim: ",new_home_goals, "Away goals: " ,  away_goals,"Sim: ", new_away_goals)
    print(f"""
    
    Score: Home {home_goals}: Away {away_goals}      XG: Home {home_shots_quality} : Away {away_shots_quality}
    Results:
    Home Wins: {home_wins}
    Draws: {draws}
    Away Wins: {away_wins}
    Average Home Goals after {minute} minute: {avg_new_home_goals}
    Average Away Goals after {minute} minute: {avg_new_away_goals}
    Average Points for Home Team: {avg_points_home:.2f}
    Average Points for Away Team: {avg_points_away:.2f}
    """)
    return home_goals, away_goals  ,avg_points_home, avg_points_away #home_wins, draws, away_wins, avg_points_home, avg_points_away

