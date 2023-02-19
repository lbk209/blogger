#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from IPython.display import Markdown
from matplotlib import pyplot as plt
import time
import functools
import os, sys, re
import numpy as np
import pandas_datareader.data as web
import pandas_datareader as pdr
import matplotlib.patches as mpatches
from io import StringIO 

from mypackage.envs import DATE_FORMAT

DATA_FORMAT_COLS = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
}

##### general utils
def add_days(strdt, days, dtf = DATE_FORMAT):
    """ return date in str in 'days' from 'strdt'
    """
    days = int(days)
    if type(strdt) is str:
        nd = datetime.strptime(strdt,dtf) + timedelta(days=days)
    elif type(strdt) is datetime:
        nd = strdt + timedelta(days=days)
    return nd.strftime(dtf)


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


@timer
def test_timer(num):
    for _ in range(num):
        sum([i**2 for i in range(10000)])


def print_bold(*strs):
    nstr = '<span style="color:blue;font-weight:bold">{}</span>'
    #nstr = '**<span style="color:blue">{}</span>**'
    strs = [str(x) for x in strs]
    display(Markdown(nstr.format(' '.join(strs))))

# TODO: print_bold와 합칠까?
def ptitle(sentence):
    print('')
    print('='*len(sentence))
    print(sentence)
    print('='*len(sentence))


# ex)
#with MeasureTime():
class MeasureTime:
    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        lapse = end_time - self.start_time
        unit = 'secs'
        if lapse > 60:
            lapse = lapse/60
            unit = 'mins'
        pstr = 'Elapsed Time: {:.2f} {:s}'.format(lapse, unit)
        print('='*len(pstr))
        print(pstr)
        print('='*len(pstr))


# ex)
#with HiddenPrints():
#   print("This will not be printed")
#print("This will be printed as before")
class HiddenPrints:
    def __init__(self, on=True, mark_end=True):
        """ if off is True, HiddenPrints doesn't work """
        self.mark_end = mark_end
        self.on = on
    def __enter__(self):
        if self.on:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.on:
            sys.stdout.close()
            sys.stdout = self._original_stdout
            # loop에서 HiddenPrints 사용하는 경우 False로 출력 남발 방지 
            #if self.mark_end: print('HiddenPrints Done.\n')
            if self.mark_end: print('-'*40, 'HiddenPrints Done.')


class PrintTest:
    def __init__(self):
        self.count = 0
    def testing(self, *args):
        self.count += 1
        print('Testing {}:'.format(self.count))
        print(*args)
        print('='*20)

    
class PrintCapture(list):
    """
    usage: ex)
        regex = 'Initial, Final, Profit: .*\d+%'
        with PrintCapturing(regex, False) as output:
            command ...
        result_list = output
    see for regex
    https://www.w3schools.com/python/python_regex.asp
    https://docs.python.org/3/howto/regex.html
    https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/07/28/regex-usage-04-intermediate/
    """
    def __init__(self, regex=None, printout=True, on=True):
        regex_dict = {
            'number': '([-+]?[\d]{1,3}(?:(?:,[\d]{3})+|[\d]*)(?:[.][\d]*)*)'
        }
        try:
            self.regex = regex_dict[regex]
        except KeyError:
            self.regex = regex
        self.output = None
        self.printout = printout
        self.on = on
    def __enter__(self):
        if self.on:
            self._stdout = sys.stdout
            sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        if self.on:
            self.output = self._stringio.getvalue()
            if self.regex is not None:
                #self.output = re.search(self.regex, self.output)
                self.output = re.findall(self.regex, self.output)
            #self.append(self.output)
            self.extend(self.output)
            del self._stringio    # free up some memory
            sys.stdout = self._stdout
            if self.printout:
                _ = [print(x) for x in self.output]
            

def end_method(plt_show=False):
    """
    plt_show: set True only if function/method has plot.
    """
    def end_method_deco(func):
        """ print line after func output """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            rv = func(*args, **kwargs)
            if plt_show: plt.show()
            print('-'*40, '{} done.'.format(func.__name__))
            return rv 
        return wrapper
    return end_method_deco


def drwa_arrow(ax, arrowstyle=None, x_tail=0.2, y_tail=0.5, x_head=0.8, y_head=0.5):
    if arrowstyle is None:
        as_str = 'Simple, head_length=0.3, head_width=0.7, tail_width=0.4'
        arrowstyle = mpatches.ArrowStyle(as_str)
    arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head),
                                 mutation_scale=100,
                                 color = 'grey',
                                 #edgecolor=None,
                                 arrowstyle = arrowstyle,
                                 alpha=0.3)
    ax.add_patch(arrow)
    ax.set_axis_off()
    return ax


def print_list_by_line(str_list, num=4):
    """ print list items by num every line """
    str_list_split = [', '.join(str_list[i:i + num]) for i in range(0, len(str_list), num)]
    print(*str_list_split, sep='\n')
    print('-'*len(str_list_split[-1]))
    
    
def twinx_lenged(ax1, ax2, colors = ['#1f77b4', '#ff7f0e']):
    """
    combine all labels to one legend
    """
    axes = (ax1, ax2)
    ### remove individual legends and add combined one
    _ = [ax.get_legend().remove() for ax in axes]
    #leg = ax1.get_lines() + ax2.get_lines()
    leg = sum([list(x.get_lines()) for x in axes], [])
    labs = [l.get_label() for l in leg]
    ax1.legend(leg, labs, loc=0)
    ### reset color for each axes
    if colors is not None:
        _ = [ax.tick_params(axis='y', labelcolor=c) for ax, c in zip(axes,colors)]
        
        
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def add_plot_with_dff_axes(fig, *tuples3, pos_ax=111, **kwargs):
    """
    add plot to fig with different x & y scales
    tuples3: ex) ('bottom','axes',-0.2), ('left','axes',-0.2)
    """
    if kwargs is None:
        ax = fig.add_subplot(pos_ax, frame_on=False)
    else:
        ax = fig.add_subplot(pos_ax, frame_on=False, **kwargs)
    for pos, ptype, amt in tuples3:
        ax.spines[pos].set_position((ptype, amt))
    make_patch_spines_invisible(ax)
    for pos, _, _ in tuples3:
        ax.spines[pos].set_visible(True)
    return ax
       
def printnb(print_out):
    if "JPY_PARENT_PID" in os.environ:
        display(print_out)
    else:
        print(print_out)

##### financial utils
testing = PrintTest().testing

def get_data_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            df, cols = func(*args, **kwargs)
        except:
            print(func.__name__,'failed')
            print('Check internet connection first')
            return
        df.index.name = 'dt' # for PlotPortfolio
        df_columns = [DATA_FORMAT_COLS[c] for c in cols]
        df.columns = df.columns.str.lower()
        df[df_columns] = df[df_columns].astype('int')

        # Fill missing columns with np.nan
        missing_columns = [col for col in df_columns if col not in df.columns]
        for missing_column in missing_columns:
            df[missing_column] = np.nan
        if len(missing_columns) > 0:
            print("Missing columns filled w/ NaN:", missing_columns)
        return df[df_columns]
    return wrapper


@get_data_decorator
def get_stock_data(symbol, start_date=None, end_date=None, source="naver", cols="c"):
    ''' load history of a stock from naver finance '''
    if symbol.startswith('A'):
        symbol = symbol[1:] # remove 1st 'A' of stock code
    if (symbol == 'kospi') or (symbol == 'kosdaq'):
        df = _get_data_yahoo(symbol, start_date, end_date, cols=cols)
    else:
        df = web.DataReader(symbol, source, start=start_date, end=end_date)
    # 기간내 증자등으로 발생한 거래중지 기간을 이전 데이터로 채움. 
    df = df.replace(to_replace=0, method='ffill')
    return df, cols


# TODO: 2021-01-05 코스피 데이터가 없음. 3천을 넘긴 기념비적 날인데.
@get_data_decorator
def _get_data_yahoo(symbol, start_date, end_date, cols='c'):
    """ symbol: 'kospi', 'kosdaq' """
    # TODO: 지저분하다. 공통 코드 변환기를 만들어 사용해야함.
    if symbol == 'kospi':
        symbol = '^KS11'
    elif symbol == 'kosdaq':
        symbol = '^KQ11'
    elif symbol.startswith('A'):
        # TODO: 종목코드면 모두 코스피로 보는데 코스닥은? 
        # 그리고 코스닥 개별 종목 이상 발견. 예) 035900.KQ (JYP) 최근 데이터 없음.
        symbol = symbol[1:] + '.KS'
    elif not symbol.endswith('.KS'):
        symbol = symbol + '.KS'
    df = pdr.get_data_yahoo(symbol, start_date, end_date)
    return df, cols
    
    
def get_data_fred(symbols, start_date=None, end_date=None):
    """
    symbols: str or list of str
     ex) currency: 'DEXKOUS', 'DEXCHUS', 'DEXJPUS'
    """
    # check for more symbols : https://fred.stlouisfed.org/categories 
    df = pdr.DataReader(symbols, 'fred', start=start_date, end=end_date)
    return df
    

def adjust_to_workdays(index, start_date, end_date, return_type='dates'):
    """ 
    reset dates to close ones in index if not business days 
    index: dataframe datetime index
    return_type: return adj. start_date & end_date if dates,
                 return index from start_date to end_date otherwise.
    """
    start_idx = index.get_loc(start_date, method='bfill')
    end_idx = index.get_loc(end_date, method='ffill')
    if return_type == 'dates':
        start_date = index[start_idx].strftime(DATE_FORMAT)
        end_date = index[end_idx].strftime(DATE_FORMAT)
        return (start_date, end_date)
    else:
        return index[start_idx:end_idx+1]
    

