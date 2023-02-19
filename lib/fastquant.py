#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 29, 2020
"""

#import functools
import pandas as pd
#import pandas_datareader.data as web
import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from fastquant import Portfolio, backtest
from datetime import datetime, timedelta
from copy import copy

from utils import HiddenPrints, end_method, get_stock_data
from utils import PrintTest
from envs import COMMISSION as TRADING_COMMISION, DATE_FORMAT

testing = PrintTest().testing # for debugging

# DONE: 현재 backtest의 plot이 잘 작동안됨. 쥬피터랩에서 표시안되고
# 백테스트 여러개 실행시 마지막 종목만 표시됨.
# 참고: https://www.backtrader.com/docu/plotting/plotting/
# => 자체 plot 로직 구현했음.
# 리밸런싱: buynhold 전략에서만 리밸런싱 구현.
# 포트폴리오 그림에서 리밸런싱일에 현금이 한번 튀어 이상해보일수 있지만
# 그냥 리밸런싱해서 주식수를 맞춘다고 생각하자. 
# 다른 전략은 리밸런싱과 무관하게 자체 매매 신호를 따름.
# (해당 전략의 매매신호를 무시하고 미리 정한 리밸런싱일이라고 
# 초기 가중치에 맞춰 매매하는 것도 모순되므로)
# TODO: backtest의 multi 전략 가능한지 검토
@end_method()
def backtest_portfolio(
    strategy, 
    stock_list, 
    start_date, 
    end_date, 
    rbal_days=0, # rebalancing only for strategy 'buynhold'
    init_cash=1000000, 
    weights=None, 
    commission=TRADING_COMMISION, 
    verbose=True, # show transctions
    sentiments=None,
    strats=None,
    plot_bp=True, # option for backtest_portfolio, not for backtest
    **kwargs_bt
):
    """Backtest portfolio with multiple holdings

    Parameters
    ----------------
    strategy : str
        strategy keys of backtest
    stock_list : str, list of str, dict
        str: 'A000150'
        list of str: ['000150','000320','000760']
        dict: {'000150': df_000150, ... }
    weights : list of float
        weights for each stock in portfolio
    best params for each strategy (opt in kospi for last 20 years): 
        {'smac':   {'fast_period': 5, 'slow_period': 60},
         'emac':   {'fast_period': 5, 'slow_period': 120},
         'rsi' :   {'rsi_period': 14, 'rsi_upper': 90, 'rsi_lower': 50},
         'bbands': {'period': 20, 'devfactor': 2.2},
         'macd':   {'fast_period': 5, 'slow_period': 60, 'signal_period': 9,
                    'sma_period': 20, 'dir_period':5}
        }
    """
    ### convert arg stock_list to symbols & stock_data
    if type(stock_list) is dict: # 종목별 주가이력 직접 입력 (for custom bt strategy)
        stock_data = copy(stock_list) # {stock1_code: stock1_history, ...}
        symbols = list(stock_list.keys())
        if len(symbols) == 1: 
            weights = [1] # 단일 종목
    else:
        if type(stock_list) is str: # 단일 종목도 사용가능. backtest와 결과 동일
            symbols = [stock_list]
            weights = [1]
        else:
            symbols = copy(stock_list)
        # 종목별 주가이력 가져오기
        stock_data = {} # {stock1_code: stock1_history, ...}
        for x in symbols:
            stock_data[x] = get_stock_data(x, start_date, end_date)

    ### get rebalancing dates
    # 해당일 휴장일수도 있으므로 datetime 인덱스로만 사용해야함. 
    # 예) stock = stock_data[c].loc[sdate:edate]
    # 실제 주가일은 주가 데이터 stock 참고. 
    rbal_dates = [] 
    sdt = datetime.strptime(start_date, DATE_FORMAT)
    edt = datetime.strptime(end_date, DATE_FORMAT)
    # rebalancing only for buynhold with rbal_days >0
    while strategy == 'buynhold':
        if rbal_days>0:
            period = timedelta(days=rbal_days)
        else:
            break
        sdt = sdt + period
        #리밸런싱일이 종료일 근처면 운용할 기간도 별로 없으니 종료
        if sdt > edt - period * 0.5:
            rbal_dates.append(edt.strftime(DATE_FORMAT)) #마지막은 리밸런싱일이 아니라 종료일
            break
        else:
            rbal_dates.append(sdt.strftime(DATE_FORMAT))
    # no rebalancing
    if not rbal_dates:
        rbal_dates.append(end_date)
        
    ### 리밸런싱 단위 기간에 주가 데이터가 있는지 확인.
    # 주가 없는 종목은 리스트에서 제외
    # 상장 폐지 등으로 중간에 주가 데이터가 없으면 backtest에서 에러 발생.
    symbols_rem = [] 
    for i, x in enumerate(symbols):
        sdate = start_date
        for edate in rbal_dates:
            stock = stock_data[x].loc[sdate:edate]
            if len(stock) > 1:
                sdate = edate
                continue
            else:
                symbols_rem.append(x)
                break
    symbols = [x for x in symbols if x not in symbols_rem]
    
    res_tot = pd.DataFrame() # 전기간 결과값
    bal = init_cash # rebalancing amout
    sdate = start_date # rebalancing date
    cdeducts = [0 for _ in symbols] # 리밸런싱에 대한 종목별 수수료 보상값
    
    ### PlotPortfolio 입력을 위한 설정
    # 수익률등 계산에 사용하지말고 그림 용도로만 사용할 것!
    pf_periodic = pd.DataFrame() # 포트폴리오 전체 종목을 합산한 일자별 평가액, 현금
    periodics = {x:pd.DataFrame() for x in symbols} # 각 종목에 대해 일자별 평가가치, 현금
    orders = {x:pd.DataFrame() for x in symbols} # 각 종목에 대해 매매 일지
    # 리밸런싱일에 주식수 변동에 의한 종목별 포트폴리오 가치 변화. 플롯의 레이블로 사용
    # 시작일과 종료일에는 변화액이 아닌 전체 액수 표시
    pfval_rebal = {x:[] for x in symbols}
    cd_dates0 = stock_data[symbols[0]].index[0]
    cd_dates = [cd_dates0] # 시작/종료일과 리밸런싱일 표시용
    
    # 각 리밸런싱일에 (최초 시작일 제외) ...
    for nth, edate in enumerate(rbal_dates):
        res_bal = pd.DataFrame() # nth 기간 종목별 결과값
        num_share = [] # num of shares for each symbol
        prices = [] # last price of share
        periodics_nth = [] # append hist['periodic'] of each backtest result of each stock
        orders_nth = [] # append hist['orders'] of each backtest result of each stock
        # 포트폴리오 각 구성 종목에 대해 ...
        for c, w in zip(symbols, weights):
            stock = stock_data[c].loc[sdate:edate]
            icash = bal*w # 리밸런싱일에 보유 현금
            # 수수료(리밸런싱 포함) 고려 최종 수익
            # see for args: ~/miniconda3/lib/python3.7/site-packages/fastquant/backtest/backtest.py
            args = [strategy, stock]
            kwargs = {'commission':commission,
                      'init_cash':icash, 
                      'return_history':True, # hist 반환용
                      'plot':False, 
                       # not working for cerebro.optstrategy(... transaction_logging=[verbose] ...)
                      'verbose':verbose, 
                      'sentiments':sentiments,
                      'strats':strats,  # Only used when strategy = "multi"
            }
            if verbose: 
                res, hist = backtest(*args, **kwargs, **kwargs_bt)
            else:
                with HiddenPrints(mark_end=False):
                #with HiddenPrints(mark_end=False, on=False): #testing
                    res, hist = backtest(*args, **kwargs, **kwargs_bt)
             
            res_bal = res_bal.append(res)
            hist_orders = hist['orders'] # trade history
            num_share.append(hist_orders['size'].sum()) # 현재 보유주식수
            prices.append(stock['close'].iloc[-1]) # 최종일 주가
            #prices.append(orders['price'].iloc[-1]) # 최종 거래일 주가
            ### (PlotPortfolio) 포트폴리오 전체 합을 구하기 쉽도록 dt, portfolio_value, cash, return 컬럼만 선택.
            # portfolio_value = cash + 보유주식평가액
            # return: pct change of portfolio_value
            ## 일자별 종목값을 합산하기 위해서 dt를 인덱스로 설정.
            # pf_periodic, periodics 구성완료되면 PlotPortfolio에서 사용을 위해 df를 컬럼으로 재설정할것
            # 가중치 0인 종목은 return이 NaN이 되어 종목간 합 계산시 역시 NaN이 되므로 0으로 변경
            periodics_nth.append(hist['periodic'].iloc[:,-4:].set_index('dt').fillna(0))
            orders_nth.append(hist_orders) # orders는 개별 종목별로만 사용됨

        ### (PlotPortfolio)
        # 현기간 매매수수료 보정하여 그래프용 종합이력, 개별종목이력 정보 생성/추가
        # DONE: 현재 로직에서 리밸런싱 시작일이 이전 periodics_nth 마지막과 이후 periodics_nth 첫날 모두 있음
        # 리밸런싱은 전날 장마감 직전 처분해 당일 장개장 직후 매매하는 것이지만,
        # 현재 종가만 관리하므로 전날 처분과 동시에 매매하는 것으로 가정.
        # 리밸런싱일 현금 그래프상가 급 증가후 감소할텐데 리밸런싱 발생으로 이해하자.
        ## 마지막날에 수수료 디덕션을 포트폴리오 가치, 현금 모두에 반영
        # (포트폴리오 가치 = 현금 + 주식평가액)
        cd_date = stock.index[-1] # not edate but the last day of stock
        for i, _ in enumerate(symbols): 
            cols = ['portfolio_value', 'cash']
            try:
                periodics_nth[i].loc[cd_date,cols] = periodics_nth[i].loc[cd_date,cols] + cdeducts[i]
            # cd_date 없는 종목은 패스 (상장폐지 등의 사유로)
            # TODO: 함수 초기에 end_date (근처) 없는 종목, 가중치(weight) 0인 종목 제외하는게 더 좋겠다.
            # (symbols, stock_data, weights 에서 해당 종목 제외)
            # 단 공휴일 등으로 end_date가 정확하지 않은 경우 어떻게 처리해야하나 고민
            except KeyError:  
                continue
        cd_dates.append(cd_date)
        # 종목별 periodics_nth을 일자별로 합산하여 이전 포트폴리오 df에 추가
        # 참고: 같은 날자 인덱스면 행갯수가 서로 다른 데이터프레임에 대한 sum 연산가능
        # (여기서는 모든 종목의 기간이 동일해 행갯수 동일)
        pf_periodic = pf_periodic.append(sum(periodics_nth))
        for i, x in enumerate(symbols): # 종목별 periodics, orders를 이전 리밸런싱 기간에 이어 추가
            if nth == 0: # 종목별 초기 투자액
                pfval_rebal[x].append(init_cash*weights[i])
            else:
                cr = periodics_nth[i].iloc[0]['portfolio_value'] - periodics[x].iloc[-1]['portfolio_value']
                pfval_rebal[x].append(cr) # 리밸런싱을 위해 추가/제외된 금액
                #pfval_rebal[x].append(periodics_nth[i].iloc[0]['portfolio_value']) # 리밸런싱후 portfolio_value
            periodics[x] = periodics[x].append(periodics_nth[i])
            orders[x] = orders[x].append(orders_nth[i])
            
        ##### 현기간 결과 보정
        # 기간내 매매수수료가 포함된 res_bal['pnl']에 리밸런싱때 수수료 디덕션을 손익에 반영.
        # 백테스트 첫 기간에는 디덕션 0
        res_bal['pnl'] = res_bal['pnl'] + cdeducts
        res_bal['final_value'] = res_bal['final_value'] + cdeducts 
        res_bal['start_date'], res_bal['end_date'] = sdate, edate
        res_bal['symbol'] = symbols
        res_tot = res_tot.append(res_bal)
        # 기간별 수익률 계산
        pnl = res_bal['pnl'].sum()
        init_cash = res_bal['init_cash'].sum() 
        ret = pnl/init_cash*100 
        print_out = '{}th period return: {:.2f}%'.format(nth+1, ret)
        pout_size = len(print_out)
        #print('-'*pout_size, print_out, '-'*pout_size, sep='\n')
        if verbose: print('-'*pout_size)
        print(print_out)
        if verbose: print('-'*pout_size)

        ##### 다음 리밸런싱 준비: cdeducts, sdate, bal, cdeducts
        if nth >= len(rbal_dates) - 1: 
            # pfval_rebal에 최종 portfolio_value 추가후 종료
            for i, x in enumerate(symbols):
                pfval_rebal[x].append(periodics_nth[i].iloc[-1]['portfolio_value'])
            break # 마지막 백테스트 완료라 다음 리밸런싱 없음
        sdate = edate # start date of next period
        bal = res_bal['final_value'].sum() # cash for next period including cdeducts
        ### commission deduction for next period:
        # 매 리밸런싱마다 보유 주식수를 무시하고 전량 매수하기 때문에
        # 실제 매수량을 고려하여 수수료를 이익/손해로 반영
        cdeducts = [] # reset current commision deducts for next period
        for w, cnum, price in zip(weights, num_share, prices):
            camt = cnum*price # 현재 보유주식수에 대한 가치
            namt = bal*w # 다음 기간의 초기 투자액
            # namt, camt 크기 예시:
            #10a > 3b: 7(10-3)주 매수하면 되는데 10주 매수. 3주(-10+3+10=3)의 수수료를 이익으로 반영
            #3a < 10b: 7주만 매도하면 되는데 3주를 매수. 4주(3-10+3=-4)의 수수료를 손실로 반영
            if namt > camt: # 리밸런싱 결과 추가 매수해야하는 경우
                cdd = commission * cnum # 기존 보유 주식수의 수수료를 이익으로 처리
            elif namt < camt: # exceeding new amount allocated for stock
                cdd = commission * (2*namt//price - cnum) # 손실 처리
            else:
                cdd = 0
            cdeducts.append(cdd)
            
    ### (PlotPortfolio)
    # 날자 인덱스를 사용하여 pf_periodic, periodics 구성완료하였으므로 원복
    # PlotPortfolio에서 사용하려면 날자(dt)는 컬럼이어야함.
    pf_periodic = pf_periodic.reset_index()
    periodics = {k:v.reset_index() for k,v in periodics.items()}

    ### 누적 수익률
    rbal_sum = res_tot.groupby(by=['start_date'])['init_cash','final_value'].sum() # 리밸런싱 기간별 구매액과 평가액
    rbal_sum['surplus'] = rbal_sum['init_cash']-rbal_sum['final_value'].shift() # 리밸런싱 기간별 구매 차익
    net_cash = rbal_sum['init_cash'].iloc[0] + rbal_sum['surplus'].sum() # 순수 구매액
    fval = rbal_sum['final_value'].iloc[-1] # 최종 평가액
    pnl = res_tot['pnl'].sum() # 총 수익
    ret = pnl / net_cash * 100 # 총 수익률
    result_summary = [net_cash, fval, ret]   
    print_out = 'Initial, Final, Profit: {:,.0f}, {:,.0f}, {:.2f}%'.format(*result_summary)
    print_out2 = 'from {} to {}'.format(start_date, end_date) 
    pout_size = len(print_out)
    print('='*pout_size, print_out, print_out2, '='*pout_size, sep='\n')

    # reorg colums of result  
    cols = res_tot.columns.to_list()
    res_tot = res_tot[cols[-2:] + [cols[7]] + cols[1:7] +cols[8:-2]]
    result_total = res_tot.set_index(['start_date','end_date','symbol'])

    #########
    ### (PlotPortfolio) plot portfolio value and trading
    # TODO: gridspec을 사용하여 ax를 정렬하고 싶은데 figsize 결정 로직이
    # value_history 안에 들어 있어 어떻게 수정해야할지 막막.    
    if plot_bp:
        num_max = 5 # max num of stocks to plot
        # sort stock_code by descending weights of more than 0
        #stock_code = [x for (y,x) in sorted(zip(weights, symbols), 
        #                                    key=lambda pair: pair[0], reverse=True
        #                                   ) if y>0]
        fsorted = {y:x for (x,y) in sorted(zip(weights, symbols), 
                                            key=lambda pair: pair[0], reverse=True
                                           ) if x>0}
        fs_stock = list(fsorted.keys())
        num_stocks = min(len(fs_stock), num_max) # num of stocks to plot
        ### 포트폴리오 전체 수익 이력 (모든 종목 합산)
        # 종목수가 2 이상일때만 전체 수익 이력 표시
        if num_stocks > 1:
            # value_history에서 입력 stock_data는 불필요하지만 x축 길이 맞추기 위해 사용
            pf_history = {'periodic':pf_periodic, 'orders':None}
            ppf = PlotPortfolio(pf_history, stock_data[fs_stock[0]])
            # plot history of total protfolio value
            ax = ppf.value_history(height=3, title='Portfolio Value', annotations=False) 
            plot_rebal_dates(ax, cd_dates[1:-1]) # 리밸런싱일 표시
            # 종목별 합
            pfval_rebal_sum = [sum(x) for x in zip(*pfval_rebal.values())]
            annotate_value_history(ax, cd_dates, pfval_rebal_sum, pfval_rebal_sum)
        ### 개별 종목 수익/주가는 num_stocks 개만 표시
        for c in fs_stock[:num_stocks]:
            ppf = PlotPortfolio({'periodic':periodics[c], 'orders':orders[c]}, 
                                stock_data[c])
            axs = ppf.history(title = '{} ({:.2f})'.format(c, fsorted[c]), annotations=False)
            plot_rebal_dates(axs[0], cd_dates[1:-1]) # 리밸런싱일 표시 (전체 시작/종료는 제외)
            annotate_value_history(axs[0], cd_dates, pfval_rebal[c], pfval_rebal[c])
        plt.show()
    #########
    return {'summary':result_summary, 'history':result_total}


# backtest_portfolio에서 리밸런싱일 표시
def plot_rebal_dates(ax, cd_dates):
    for x in cd_dates: 
        ax.axvline(x, c='grey', ls='--') 


def annotate_value_history(ax, date, pfval, annotation):
    """ 
    date: 시작일, 리밸런싱일..., 종료일 리스트
    pfval: 시작금액, 리밸런싱 추가/감소액, 종료일 평가액 리스트
    """
    ### fisrt day: 초기금액 표시
    # 'va':'top' 으로 해야 bottom으로 표시됨
    kwargs = {'xycoords':'data', 'ha':'left', 'va':'top'}
    kwargs.update({'bbox':dict(facecolor='white', edgecolor='white', alpha=0.3)})
    ax.annotate('{:^,.0f}'.format(annotation[0]), (date[0],pfval[0]), **kwargs)
    ### rebalancing days: 전단계 대비 리밸런싱 증가/감소분 표시
    icash = pfval[0] # initial cash
    tcash = pfval[0] # icash with extra cash for rebalancing
    for x, y, z in zip(date[1:-1], pfval[1:-1], annotation[1:-1]):
        tcash += y
        kwargs.update({'ha':'left'})
        # 절대값이 0.1 보다 작으면 0으로 본다.
        if z > 0.1:
           anstr = '(+ {:^,.0f})'
        elif z < -0.1:
           anstr = '(- {:^,.0f})'
        else:
           anstr = '({:^,.0f})'
        # (x, tcash) 에 출력하면 tight subplots을 벗어나 (x, icash)로 일정하게 설정
        ax.annotate(anstr.format(abs(z)), (x, icash), **kwargs)
    ### last day: 최종액 표시
    if pfval[-1] > tcash:
        color = 'blue'
    elif pfval[-1] < tcash:
        color = 'red'
    else:
        color = 'black'
    #kwargs.update({'ha':'right', 'color':color})
    #ax.annotate('{:^,.0f}'.format(annotation[-1]), (date[-1],pfval[-1]), **kwargs)
    kwargs.update({'ha':'center', 'color':color})
    kwargs.update({'bbox':dict(facecolor='white', edgecolor='white', alpha=0.7)})
    ret = (annotation[-1] / tcash - 1) * 100
    ax.annotate('{:^,.0f}\n( {:^.0f}% )'.format(annotation[-1], ret), (date[-1],pfval[-1]), **kwargs)


# TODO: 고유 포트폴리오 추가. see p467~ 핸즈온 머신러닝, 립러닝 알고리즘 트레이딩
class Portfolio(Portfolio):
    # super
    #~/miniconda3/lib/python3.7/site-packages/fastquant/portfolio.py
    def __init__(self, model, *args, **kwargs):
        """ model = 'minvar', 'maxdiv', 'sharpe' """
        self.model = model
        super().__init__(*args, **kwargs)

    def get_data(self):
        dfs = []
        for i in self.stock_list:
            # get data from naver finance
            df = get_stock_data(i, self.start_date, self.end_date)
            df.columns = [i]
            dfs.append(df)
        data = pd.concat(dfs, axis=1)
        data.index.name = "DATE"
        return data

    # minimum variance portfolio
    def min_func_covar(self, weights):
        ret = self.returns
        # standardize returns for robust solution
        ret = (ret - ret.mean()) / ret.std() 
        #np.dot(weights.T, np.dot(self.returns.cov() * 252, weights))
        covmat = np.dot(weights.T, np.dot(ret.cov(), weights))
        return covmat

    # maximize diversification ratio
    def max_func_dratio(self, weights):
        # consistent solution even if no standardization
        portfolio_var = np.sqrt(
            np.dot(weights.T, np.dot(self.returns.cov(), weights))
        )
        var_psum = np.dot(self.returns.cov().to_numpy().diagonal(), weights)
        dratio = var_psum / portfolio_var
        # we want to maximize div ratio = minimize the negative of it
        return -dratio

    def get_objective(self):
        if self.model == 'sharpe':
            objfun = self.min_func_sharpe
        elif self.model == 'minvar':
            objfun = self.min_func_covar
        elif self.model == 'maxdiv':
            objfun = self.max_func_dratio
        return objfun

    def optimize_portfolio(self):
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(len(self.stock_list)))
        init_weights = (
            self.random_weights
            if self.optimum_weights is None
            else self.optimum_weights
        )
        optimum = optimization.minimize(
            fun=self.get_objective(),
            x0=init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if optimum.success:
            self.optimum_weights = optimum["x"].round(3)
            self.print_optimal_portfolio()
            return optimum
        else:
            print("Optimization failed. Try different init_weights.")


class PlotPortfolio():
    """ fastquant.backtest의 history result와 주가(종가)를 입력으로
        포트폴리오, 매매, 주가 이력을 그리기 """
    def __init__(self, history_portfolio, history_stock=None,
                 width=10, height=5, vratio=0.3):
        """ history_portfolio = {'orders': df1, 'preiodic':df2} 
            2nd output of backtest(return_history = True)
        """
        self.width = width
        self.height = height
        self.vratio = vratio
        self.history_portfolio = history_portfolio # history return of backtest()
        self.value_fsize = (width, height*vratio)
        self.orders_fsize = (width, height*(1-vratio))
        #self.height_ratios = [vratio, 1-vratio]
        #self.history_fsize = (width, height) # set in history method instead
        if history_stock is not None: # axis x range. convert index to dt(datetime)
            self.history_stock = history_stock.rename_axis('dt') # index 이름은 항상 'dt'
            self.x_dt = self.history_stock['close'].reset_index()[['dt']] 
        else: 
            self.history_stock = None
            self.x_dt = None
        
    def value_history(self, ax=None, title='', height=None, annotations=True):
        """ plot portfolio value and cash history 
        annotations: 초기자금과 최종 자산가치를 그래프에 표시할지를 결정.
                     표시 내용을 변경하거나 더 추가하려면 None으로 설정하고
                     ax를 받아 작업하면됨.
        """ 
        if ax is None:
            if height is not None:
                figsize = (self.width, height)
            else:
                figsize = self.value_fsize 
            _, ax = plt.subplots(1,1,figsize=figsize)
        
        # periodic: df of columns dt(time), portfolio_value, cash, return
        periodic = self.history_portfolio['periodic']
        # 주가 일자와 매매 시작/종료 일자 사이 간격 그래프 채우기
        if self.x_dt is not None:
            periodic = periodic.append(self.x_dt[:1], ignore_index=True)
            periodic = periodic.append(self.x_dt[-1:], ignore_index=True)
            periodic = periodic.sort_values(by='dt')
            periodic = periodic.fillna(method='bfill')
            periodic = periodic.fillna(method='ffill')
        # 현금 & 평가가치 그리기
        # named colors: https://matplotlib.org/3.3.2/gallery/color/named_colors.html
        periodic.plot(x='dt', y='cash', ax=ax, c='darkseagreen')
        periodic.plot(x='dt', y='portfolio_value', ax=ax, c='darkorange')
        ax.legend(framealpha=0.5, edgecolor='white', loc=3)
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(title)
        if annotations: # 시작금액과 최종평가액 표시
            pfval_start = periodic['portfolio_value'].iloc[0]
            dt_start = periodic['dt'].iloc[0]
            pfval_end = periodic['portfolio_value'].iloc[-1]
            dt_end = periodic['dt'].iloc[-1]
            ### fisrt day: 초기금액 표시
            kwargs = {'xycoords':'data', 'ha':'left', 'va':'top'}
            kwargs.update({'bbox':dict(facecolor='white', edgecolor='white', alpha=0.3)})
            ax.annotate('{:^,.0f}'.format(pfval_start), (dt_start,pfval_start), **kwargs)
            ### last day: 최종액 표시
            if pfval_end > pfval_start:
                color = 'blue'
            elif pfval_end < pfval_start:
                color = 'red'
            else:
                color = 'black'
            kwargs.update({'ha':'center', 'color':color})
            kwargs.update({'bbox':dict(facecolor='white', edgecolor='white', alpha=0.7)})
            ret = (pfval_end / pfval_start - 1) * 100
            ax.annotate('{:^,.0f}\n( {:^.0f}% )'.format(pfval_end, ret), (dt_end,pfval_end), **kwargs)
        return ax

    def orders_history(self, history_stock=None, ax=None, title=''):
        """ 
        plot price and trading histories 
        history_stock must have 'close' column
        """
        # orders: df of columns dt(time), type(buy/sell), price
        orders = self.history_portfolio['orders']
        if ax is None: _, ax = plt.subplots(1,1,figsize=self.orders_fsize)
        ### plot close price if exists
        if history_stock is None:
            if self.history_stock is not None:
               close = self.history_stock['close']
               ax.plot(close, linewidth=1.5, color='grey')
        else:
            close = history_stock['close']
            ax.plot(close, linewidth=1.5, color='grey')
            if self.history_stock is None:
                self.history_stock = history_stock.rename_axis('dt')
                self.x_dt = close.reset_index()[['dt']] # update for value history
        ### plot buy and sell history
        kwargs = {'x':'dt', 'y':'price', 'kind':'scatter'}
        kwargs2 = {'buy':{'label':'buy', 'marker':10, 'c':'blue', 's':100}}
        kwargs2['sell'] = {'label':'sell', 'marker':11, 'c':'red', 's':100}
        for x in kwargs2.keys():
            trade = orders[orders['type']==x]
            trade.plot(ax=ax, **kwargs, **kwargs2[x])
        ax.legend(framealpha=0.5, edgecolor='white')
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(title)
        return ax

    def history(self, history_stock=None, indicator=False, 
                      title='Portfolio value & Trading', annotations=True):
        """
        plot portfolio value history and trading history 
        history_stock columns except for 'close' & 'custom' 
            are to be plotted if indicator true
        """
        if history_stock is None: history_stock = self.history_stock
        periodic = self.history_portfolio['periodic']
        orders = self.history_portfolio['orders']
        if indicator:
            rc = [3,1]
            height = self.height*(1+self.vratio)
            figsize = (self.width, height)
            height_ratios = [self.vratio, 1-self.vratio, self.vratio]
        else:
            rc = [2,1]
            height = self.height
            figsize = (self.width, height)
            height_ratios = [self.vratio, 1-self.vratio]
        fig, axs = plt.subplots(*rc,figsize=figsize, sharex=True,
                                gridspec_kw={'height_ratios':height_ratios, 'hspace':0})
        # self.x_dt 갱신을 위해 self.orders_history 먼저 수행
        _ = self.orders_history(history_stock, ax=axs[1]) # scatter orders
        if indicator:
            # 'custom' is reserved for custom trading strategy of backtest
            cols = [x for x in history_stock.columns if x not in ['close','custom']]
            _ = history_stock[cols].plot(ax=axs[2]) # plot indicator
            axs[2].set_xlabel('')
            axs[2].set_ylabel('')
        # run self.value_history last for autoscaling
        _ = self.value_history(ax=axs[0], annotations=annotations) # plot value history of portfolio
        fig.autofmt_xdate() #rotate date ticklabels and right align them
        fig.suptitle(title, y=0.92)
        return axs



