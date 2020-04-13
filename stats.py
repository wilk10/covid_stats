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
    CHART_TYPES = ['cumul', 'log_confirmed', 'log_active']
    DROP_COLUMNS = ['days_to_0', 'date_when_0_reached']
    MIN_FOR_CHARTS = 1000

    def __init__(self):
        self.cwd = pathlib.Path.cwd()
        self.df_by_status = self.make_df_by_status()
        self.countries = self.df_by_status['confirmed'].country.unique()
        self.drop_df_dir = self.cwd / 'drop_df'
        self.dir_by_chart_type = self.make_all_dirs()
        self.last_day_file = self.cwd / "last_day.json"
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
        if country == 'china' and province == 'hong kong':
            country_series = pandas.Series(['hong kong', ''], index=['country', 'province'])
        else:
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

    def make_all_dirs(self):
        if not self.drop_df_dir.exists():
            self.drop_df_dir.mkdir()
        dir_by_chart_type = dict.fromkeys(self.CHART_TYPES)
        charts_dir = self.cwd / 'charts'
        if not charts_dir.exists():
            charts_dir.mkdir()
        for chart_type in self.CHART_TYPES:
            chart_type_dir = charts_dir / chart_type
            if not chart_type_dir.exists():
                chart_type_dir.mkdir()
            dir_by_chart_type[chart_type] = chart_type_dir
        return dir_by_chart_type

    @staticmethod
    def shorten_index_dates(df):
        datetimes = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in df.index]
        short_dates = numpy.array([d.strftime('%d/%m') for d in datetimes])
        df.index = short_dates
        return df

    def prepare_country_df(self, country):
        df = pandas.DataFrame(columns=self.STATUS)
        for status, status_df in self.df_by_status.items():
            country_status_df = status_df.loc[status_df.country == country]
            sum_df = country_status_df.groupby('country').sum()
            df[status] = sum_df.iloc[0, :]
        df = df.fillna(method='ffill')
        df['active'] = df.confirmed - df.deaths - df.recovered
        df = self.shorten_index_dates(df)
        return df

    def make_cumulative_chart(self, df, country):
        df = df[['active', 'deaths', 'recovered']]
        df = df.loc[(df.T != 0).any()]
        index_ticks = numpy.arange(0, len(df), 5)
        label_ticks = df.index[index_ticks]
        colours = [f'tab:{col}' for col in ['orange', 'red', 'blue']]
        df.plot(kind='bar', color=colours, stacked=True, width=0.84)
        plt.xticks(index_ticks, label_ticks)
        plt.legend(loc='upper left')
        plt.title(country)
        plt.grid()
        plot_path = self.dir_by_chart_type['cumul'] / f'{country}.png'
        if not plot_path.exists():
            print(f'new charts for {country}')
        plt.savefig(plot_path)
        plt.close()

    @staticmethod
    def fit_function(x, a, b):
        return numpy.exp(a * x + b)

    def add_exponential_fits(self, df, statuses):
        drop_series = None
        last_10_days_df = df.iloc[-10:, :]
        for column in statuses:
            params, _ = scipy.optimize.curve_fit(
                self.fit_function,
                range(len(df)-10, len(df)),
                last_10_days_df[column],
                method='dogbox'
            )
            data_fit = self.fit_function(range(len(df)), *params)
            df[f'{column}_line'] = data_fit
            a, b = params
            if 'active' in statuses and a < 0:
                days_to_zero = -b // a
                zero_day = self.today + datetime.timedelta(days=days_to_zero)
                zero_day = zero_day.strftime('%Y-%m-%d')
                drop_series = pandas.Series([days_to_zero, zero_day], index=self.DROP_COLUMNS)
        return df, drop_series

    def make_log_chart(self, df, country, status, drop_df):
        statuses = [status, 'deaths'] if status == 'confirmed' else [status]
        df = df[statuses]
        df = df.loc[df[status] > 10]
        pow10 = numpy.array([10**i for i in range(0, 10)])
        mask = [df[status].max() / value > 0.1 for value in pow10]
        ticks = pow10[mask]
        updated_df, drop_series = self.add_exponential_fits(df, statuses)
        if drop_series is not None:
            drop_series.name = country
            drop_df = drop_df.append(drop_series)
        if status == 'confirmed':
            colors = ['tab:blue', 'tab:red'] * 2
            line_styles = ['-', '-', '--', '--']
        else:
            colors = ['tab:orange']
            line_styles = ['-', '--']
        updated_df.plot(color=colors, style=line_styles)
        plt.yscale('log')
        plt.yticks(ticks, [str(tick) for tick in ticks])
        plt.ylim(10, ticks[-1])
        plt.title(country)
        plt.grid(which='both')
        plot_path = self.dir_by_chart_type[f'log_{status}'] / f'{country}.png'
        plt.savefig(plot_path)
        plt.close()
        return drop_df

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
        active_df = active_df.sort_values('avg_growth', ascending=False)
        return active_df

    def save_last_day_and_drop_df(self, drop_df):
        today_str = self.today.strftime('%Y-%m-%d')
        new_last_day = {'last_day': today_str}
        with open(str(self.last_day_file), 'w') as f:
            json.dump(new_last_day, f)
        drop_df_path = self.drop_df_dir / f'{today_str}.csv'
        drop_df.to_csv(str(drop_df_path))

    def make_drop_chart(self):
        chart_df = pandas.DataFrame()
        for file in self.drop_df_dir.iterdir():
            day_df = pandas.read_csv(str(file), index_col=0)
            day_row = day_df.days_to_0.loc[day_df.days_to_0 <= 365*2]
            day_row.name = file.name.split('.')[0]
            chart_df = chart_df.append(day_row)
        chart_df = self.shorten_index_dates(chart_df)
        chart_df.plot(marker='o', markersize=3)
        ticks = numpy.arange(0, 365*2+1, 365//4)
        ticks_labels = [str(n) for n in numpy.arange(0, 25, 3)]
        plt.yticks(ticks, ticks_labels)
        plt.ylim(0, ticks[-1])
        plt.grid(which='both')
        plt.legend()
        plt.ylabel('months from now')
        plt.xlabel('day of prediction')
        plt.title('months to 0 active cases')
        plot_path = self.cwd / 'charts' / 'drop_df.png'
        plt.savefig(plot_path)
        plt.close()

    def run(self):
        if self.today > self.last_day:
            active_df = pandas.DataFrame()
            drop_df = pandas.DataFrame(columns=self.DROP_COLUMNS)
            for country in self.countries:
                country_df = self.prepare_country_df(country)
                if country_df.confirmed.iloc[-1] > self.MIN_FOR_CHARTS:
                    active_df = self.make_active_df(country_df.active, country, active_df)
                    self.make_cumulative_chart(country_df, country)
                    for status in ['confirmed', 'active']:
                        drop_df = self.make_log_chart(country_df, country, status, drop_df)
            self.save_last_day_and_drop_df(drop_df)
            print(active_df.to_string())
            print(f'\ncountries with dropping active cases:\n{drop_df}')
            self.make_drop_chart()

    def make_past_drop_dfs(self, start_date_str):
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
        days_in_between = (self.today - start_date).days
        dates = [start_date + datetime.timedelta(days=i) for i in range(days_in_between)]
        for considered_date in dates:
            drop_df = pandas.DataFrame(columns=self.DROP_COLUMNS)
            for country in self.countries:
                df = self.prepare_country_df(country)
                datetimes = [datetime.datetime.strptime(d + '/2020', '%d/%m/%Y') for d in df.index]
                filtered_datetimes = [d for d in datetimes if d.date() < considered_date]
                filtered_rows = [d.strftime('%d/%m') for d in filtered_datetimes]
                filtered_df = df.loc[filtered_rows, :]
                if filtered_df.confirmed.iloc[-1] > self.MIN_FOR_CHARTS:
                    _, drop_series = self.add_exponential_fits(filtered_df, ['active'])
                    if drop_series is not None:
                        drop_series.name = country
                        drop_df = drop_df.append(drop_series)
            date_str = considered_date.strftime('%Y-%m-%d')
            drop_df_path = self.drop_df_dir / f'{date_str}.csv'
            drop_df.to_csv(str(drop_df_path))
        self.make_drop_chart()


if __name__ == '__main__':
    CovidStats().run()
    #CovidStats().make_past_drop_dfs('2020-03-01')
