from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

class FootballMachtPredictor():

    def __init__(self):
        """
        
        """
        self.model = load_model('../model/model_match_result')
        
        historical_data = pd.read_csv('../data/historical_data.csv')
        historical_data.drop(columns=['Unnamed: 0'], inplace=True) #drop columns we won't use
        historical_data['date'] = pd.to_datetime(historical_data['date'], infer_datetime_format=True) #convert date to datetime format
        historical_data['goalDiff'] = historical_data.apply(lambda row: row['localGoals'] - row['visitorGoals'], axis=1) #convention: +localGoals -visitorGoals 
        historical_data['result'] = 'draw'
        historical_data.loc[historical_data['goalDiff'] > 0, 'result'] = 'local'
        historical_data.loc[historical_data['goalDiff'] < 0, 'result'] = 'visitor'
        self.historical_data = historical_data

    def get_season(self, date):
        """
        This function returns the season in correspondance to the given date
        """
        dt = pd.to_datetime(date, format="%Y/%m/%d")
        if dt.month > 6:
            season = str(dt.year)+'-'+str(dt.year+1)[-2:]
        else:
            season = str(dt.year-1)+'-'+str(dt.year)[-2:]
        return season

    def preprocessing_history(self, team_a, team_b, date):
        """
        This function computes the input vector for the model. All calculations are made up to "date"
        ----------
        parameters
            team_a: local team of the match
            team_b: visitor team of the match
            date: date of the match
        -------
        returns
            array: [
            A_WR: historical team_a's winrate playing as local team
            B_WR: historical team_b's winrate playing as visitor team
            A_GD: historical team_a's goal difference
            B_GD: historical team_b's goal difference
            MU: historical match up -> team_a's wins against team_b -("minus") team_b's wins against team_a
            A_current_season_WR: team_a's win rate the in the current season
            B_current_season_WR: team_b's win rate the in the current season
            ] 
        """
        HISTORICAL_DATA = self.historical_data
        dt = pd.to_datetime(date, format="%Y/%m/%d")
        df_date = HISTORICAL_DATA[HISTORICAL_DATA['date']<dt] #data up to date

        #Historical win-rates
        df_A = df_date[df_date['localTeam'].eq(team_a)]
        try:
            A_WR = sum(df_A['result_local'])/len(df_A)
        except:
            A_WR = 0

        df_B = df_date[df_date['visitorTeam'].eq(team_b)]
        try:
            B_WR = sum(df_B['result_visitor'])/len(df_B)
        except:
            B_WR = 0

        #Goal difference
        A_GD = sum(df_A['goalDiff'])
        B_GD = sum(df_B['goalDiff'])

        #Match up history
        df_match = df_date[df_date['localTeam'].eq(team_a) & df_date['visitorTeam'].eq(team_b)]
        MU = sum(df_match['result_local']) - sum(df_match['result_visitor'])

        #Season win-rates
        season = self.get_season(date)
        df_season = df_date[df_date['season'].eq(season)]

        df_A_local = df_season[df_season['localTeam'].eq(team_a)]
        df_A_visitor = df_season[df_season['visitorTeam'].eq(team_a)]
        try:
            A_current_season_WR = (sum(df_A_local['result_local']) + sum(df_A_visitor['result_visitor']))/max(max(df_A_local['round']), max(df_A_visitor['round']))
        except:
            A_current_season_WR = 0
        df_B_local = df_season[df_season['localTeam'].eq(team_b)]
        df_B_visitor = df_season[df_season['visitorTeam'].eq(team_b)]
        try:
            B_current_season_WR = (sum(df_B_local['result_local']) + sum(df_B_visitor['result_visitor']))/max(max(df_B_local['round']), max(df_B_visitor['round']))
        except:
            B_current_season_WR = 0

        return np.array([A_WR, B_WR, A_GD, B_GD, MU, A_current_season_WR, B_current_season_WR])

    def get_prev_season(self, date):
        """
        This function returns the pevious season in correspondance to the given date
        """
        dt = pd.to_datetime(date, format="%Y/%m/%d")
        if dt.year == 1970:
            season = '1970-71'
        else:
            if dt.month > 6:
                season = str(dt.year-1)+'-'+str(dt.year)[-2:]
            else:
                season = str(dt.year-2)+'-'+str(dt.year-1)[-2:]
        return season

    def preprocessing_N(self, team_a, team_b, date, N=10):
        """
        This function computes the input vector for the model. All calculations are made up to "date"
        ----------
        parameters
            team_a: local team of the match
            team_b: visitor team of the match
            date: date of the match
        -------
        returns
            array: [
            A_WR_N: team_a's winrate playing as local team in the last N matches
            B_WR_N: team_b's winrate playing as visitor team in the last N matches
            A_GD_N: team_a's goal difference in the last N matches
            B_GD_N: team_b's goal difference in the last N matches
            MU_N: match up in the last N matches -> team_a's wins against team_b -("minus") team_b's wins against team_a (in the last N matches)
            A_prev_season_GD: team_a's goal difference the in the previous season
            B_prev_season_GD: team_b's goal difference the in the previous season
            ] 
        """
        HISTORICAL_DATA = self.historical_data
        dt = pd.to_datetime(date, format="%Y/%m/%d")
        df_date = HISTORICAL_DATA[HISTORICAL_DATA['date']<dt] #data up to date

        #N days before win-rates
        df_A = df_date[df_date['localTeam'].eq(team_a)]
        df_A = df_A.tail(N)
        try:
            A_WR_N = sum(df_A['result_local'])/len(df_A)
        except:
            A_WR_N = 0

        df_B = df_date[df_date['visitorTeam'].eq(team_b)]
        df_B = df_B.tail(N)
        try:
            B_WR_N = sum(df_B['result_visitor'])/len(df_B)
        except:
            B_WR_N = 0

        #N days before goal difference
        A_GD_N = sum(df_A['goalDiff'])
        B_GD_N = sum(df_B['goalDiff'])

        #N days before match up history
        df_match = df_date[df_date['localTeam'].eq(team_a) & df_date['visitorTeam'].eq(team_b)]
        df_match = df_match.tail(N)
        MU_N = sum(df_match['result_local']) - sum(df_match['result_visitor'])

        #Previous season win-rates
        prev_season = self.get_prev_season(date)
        df_prev_season = df_date[df_date['season'].eq(prev_season)]

        df_A_prev_season = df_prev_season[df_prev_season['localTeam'].eq(team_a)]
        A_prev_season_GD = sum(df_A_prev_season['goalDiff'])
        
        df_B_prev_season = df_prev_season[df_prev_season['localTeam'].eq(team_b)]
        B_prev_season_GD = sum(df_B_prev_season['goalDiff'])

        return np.array([A_WR_N, B_WR_N, A_GD_N, B_GD_N, MU_N, A_prev_season_GD, B_prev_season_GD])

    def get_response(self, input_data: dict):
        """
        
        """
        local = input_data['local']
        visitor = input_data['visitor']
        date = input_data['date']

        x_history = self.preprocessing_history(local, visitor, date)
        x_10 = self.preprocessing_N(local, visitor, date)

        x = np.append(x_history, x_10)

        confidence = np.max(self.model.predict(x.reshape(1, -1)))
        label_idx = np.argmax(self.model.predict(x.reshape(1, -1)))
        labels = ['draw', 'local', 'visitor']
        response = {'winner': labels[label_idx], 'confidence': confidence}
        
        return response

