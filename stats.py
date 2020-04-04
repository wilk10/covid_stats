import pandas
import requests
import datetime
import pathlib
import numpy
import json
import scipy.optimize
import matplotlib.pyplot as plt


class CovidStats:
    URL = 'https://coronavirus-tracker-api.herokuapp.com'
    STATUS = ['confirmed', 'deaths', 'recovered']
    MIN_FOR_CHARTS = 1000

    def __init__(self):
        cwd = pathlib.Path.cwd()
        self.df_by_status = self.make_df_by_status()
        self.countries = self.df_by_status['confirmed'].country.unique()
        self.cumul_dir = cwd / "cumul_charts"
        if not self.cumul_dir.exists():
            self.cumul_dir.mkdir()
        self.log_dir = cwd / "log_charts"
        if not self.log_dir.exists():
            self.log_dir.mkdir()
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

    def prepare_country_df(self, country):
        df = pandas.DataFrame(columns=self.STATUS)
        for status, status_df in self.df_by_status.items():
            country_status_df = status_df.loc[status_df.country == country]
            sum_df = country_status_df.groupby('country').sum()
            df[status] = sum_df.iloc[0, :]
        df = df.fillna(method='ffill')
        df['active'] = df.confirmed - df.deaths - df.recovered
        datetimes = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in df.index]
        short_dates = numpy.array([d.strftime('%d/%m') for d in datetimes])
        df.index = short_dates
        return df

    def make_cumulative_chart(self, df, country):
        df = df[['active', 'deaths', 'recovered']]
        df = df.loc[(df.T != 0).any()]
        index_ticks = numpy.arange(0, len(df), 5)
        label_ticks = df.index[index_ticks]
        df.plot(kind='bar', color=['orange', 'red', 'dodgerblue'], stacked=True, width=0.84)
        plt.xticks(index_ticks, label_ticks)
        plt.legend(loc='upper left')
        plt.title(country)
        plt.grid()
        plot_path = self.cumul_dir / f'{country}.png'
        if not plot_path.exists():
            print(f'new cumulative chart for {country}')
        plt.savefig(plot_path)
        plt.close()

    @staticmethod
    def fit_function(x, a, b):
        return numpy.exp(a * x + b)

    def add_exponential_fits(self, df):
        last_10_days_df = df.iloc[-10:, :]
        for column in self.STATUS[:2]:
            params, _ = scipy.optimize.curve_fit(
                self.fit_function,
                range(len(df)-10, len(df)),
                last_10_days_df[column],
                method='dogbox'
            )
            data_fit = self.fit_function(range(len(df)), *params)
            df[f'{column}_line'] = data_fit
        return df

    def make_log_chart(self, df, country):
        df = df[self.STATUS[:2]]
        df = df.loc[df.confirmed > 10]
        pow10 = numpy.array([10**i for i in range(0, 10)])
        mask = [df.confirmed.max() / value > 0.1 for value in pow10]
        ticks = pow10[mask]
        updated_df = self.add_exponential_fits(df)
        color_map = plt.get_cmap("tab10")
        colors = [color_map(0), color_map(1)] * 2
        line_styles = ['-', '-', '--', '--']
        updated_df.plot(color=colors, style=line_styles)
        plt.yscale('log')
        plt.yticks(ticks, [str(tick) for tick in ticks])
        plt.ylim(10, ticks[-1])
        plt.title(country)
        plt.grid(which='both')
        plot_path = self.log_dir / f'{country}.png'
        if not plot_path.exists():
            print(f'new log chart for {country}')
        plt.savefig(plot_path)
        plt.close()

    @staticmethod
    def make_active_df(actives, country, active_df):
        last_day_data = actives.iloc[-1]
        last_actives = actives.iloc[-4:]
        new_old_actives = zip(last_actives.iloc[1:], last_actives[:-1])
        growths = [(new - old) / old for new, old in new_old_actives]
        avg_growth = numpy.mean(growths)
        new_active = int(last_day_data * avg_growth)
        data = [last_day_data, avg_growth * 100, new_active, last_day_data + new_active]
        columns = ['active', 'avg_growth', 'new', 'next_day']
        data_series = pandas.Series(data, index=columns, name=country)
        active_df = active_df.append(data_series)
        active_df = active_df.sort_values('active', ascending=False)
        return active_df

    def save_last_day(self):
        new_last_day = {'last_day': self.today.strftime('%Y-%m-%d')}
        with open(str(self.last_day_file), 'w') as f:
            json.dump(new_last_day, f)

    def run(self):
        if self.today > self.last_day:
            active_df = pandas.DataFrame()
            for country in self.countries:
                country_df = self.prepare_country_df(country)
                if country_df.confirmed.iloc[-1] > self.MIN_FOR_CHARTS:
                    active_df = self.make_active_df(country_df.active, country, active_df)
                    self.make_cumulative_chart(country_df, country)
                    self.make_log_chart(country_df, country)
            self.save_last_day()
            print(active_df.to_string())


if __name__ == '__main__':
    CovidStats().run()
