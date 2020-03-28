import pandas
import requests
import datetime
import pathlib
import numpy
import json
import matplotlib.pyplot as plt


class CovidStats:
    URL = 'https://coronavirus-tracker-api.herokuapp.com'
    STATUS = ['confirmed', 'deaths', 'recovered']
    MIN_FOR_CUMUL_CHART = 1000

    def __init__(self):
        cwd = pathlib.Path.cwd()
        self.df_by_status = self.make_df_by_status()
        self.countries = self.df_by_status['confirmed'].country.unique()
        self.cumul_dir = cwd / "cumul_charts"
        if not self.cumul_dir.exists():
            self.cumul_dir.mkdir()
        self.last_day_file = cwd / "last_day.json"
        self.last_day = self.get_last_day()
        self.today = datetime.date.today()

    def get_last_day(self):
        with open(str(self.last_day_file)) as f:
            last_day = json.load(f)
        last_day_date = datetime.datetime.strptime(last_day['last_day'], '%Y-%m-%d').date()
        return last_day_date

    @staticmethod
    def get_data_series(country_data):
        history = country_data['history']
        parsed_history = {
            datetime.datetime.strptime(date, '%m/%d/%y'): value
            for date, value in history.items()
        }
        history_series = pandas.Series(parsed_history, index=sorted(parsed_history.keys()))
        sorted_dates = [date.strftime('%Y-%m-%d') for date in sorted(parsed_history.keys())]
        history_series = history_series.reindex(sorted_dates)
        country = country_data['country'].lower()
        province = country_data['province'].lower()
        country_series = pandas.Series([country, province], index=['country', 'province'])
        country_series = country_series.append(history_series)
        return country_series

    def make_df_by_status(self):
        df_by_status = {s: pandas.DataFrame() for s in self.STATUS}
        for status in self.STATUS:
            response = requests.get(f'{self.URL}/{status}')
            locations_data = response.json()['locations']
            for country_data in locations_data:
                country_series = self.get_data_series(country_data)
                df_by_status[status] = df_by_status[status].append(
                    country_series, ignore_index=True
                )
            print(f'{status} df made')
        return df_by_status

    def make_chart(self, df, country):
        df = df.drop('confirmed', axis=1)
        df = df[['deaths', 'active', 'recovered']]
        df = df.loc[(df.T != 0).any()]
        datetimes = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in df.index]
        short_dates = numpy.array([d.strftime('%d/%m') for d in datetimes])
        df.reindex(short_dates)
        index_ticks = numpy.arange(0, len(df), 5)
        label_ticks = short_dates[index_ticks]
        df.plot(kind='bar', color=['red', 'orange', 'dodgerblue'], stacked=True, width=0.84)
        plt.xticks(index_ticks, label_ticks)
        plt.legend(loc='upper left')
        plt.title(country)
        plt.grid()
        plot_path = self.cumul_dir / f'{country}.png'
        if not plot_path.exists():
            print(f'new chart for {country}')
        plt.savefig(plot_path)
        plt.close()

    def save_last_day(self):
        new_last_day = {'last_day': self.today.strftime('%Y-%m-%d')}
        with open(str(self.last_day_file), 'w') as f:
            json.dump(new_last_day, f)

    def make_cumulative_charts(self):
        for country in self.countries:
            df = pandas.DataFrame(columns=self.STATUS)
            for status, status_df in self.df_by_status.items():
                country_status_df = status_df.loc[status_df.country == country]
                sum_df = country_status_df.groupby('country').sum()
                df[status] = sum_df.iloc[0, :]
            df = df.fillna(method='ffill')
            # df['confirmed_new'] = df.confirmed.diff().fillna(0)
            df['active'] = df.confirmed - df.deaths - df.recovered
            if df.confirmed.iloc[-1] > self.MIN_FOR_CUMUL_CHART:
                if self.today > self.last_day:
                    self.make_chart(df, country)
        self.save_last_day()

    def run(self):
        self.make_cumulative_charts()


if __name__ == '__main__':
    CovidStats().run()
