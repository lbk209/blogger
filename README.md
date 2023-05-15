# <img style="vertical-align: bottom;" height="45" src="camel_laptop_icon.png"/> [데이터_가내수공업](https://leebaekku.blogspot.com)
[블로그](https://leebaekku.blogspot.com) 포스팅을 작성하기 위해 사용한 주피터 노트북 리스트
* [백테스팅 (Backtesting)](backtesting_scenarios.ipynb) 
  - 코스피 지수에 대하여 매매 전략별(buy & hold, MACD, Bollinger bands), 분할 매수 횟수별, 보유 기간별 수익률 비교.
  - 베이즈 추정(Bayesian estimation)으로 매매 전략별 모집단의 수익률을 추정.
  - 백테스팅은 [fastquant](https://github.com/enzoampil/fastquant), 베이즈 추정은 [PyMC](https://www.pymc.io) 패키지 사용.
  - 관련 포스팅: [주식 투자의 모의고사 백테스팅 Backtesting](https://leebaekku.blogspot.com/2023/03/12-backtesting.html)
* [설문조사 통계분석](survey2.ipynb)
  - 연애·결혼 설문조사의 설문 항목별 표본 오차 계산.
  - 성별·세대와 설문 항목의 상관 관계 분석.
  - 관련 포스팅: [20대도 연애하고 결혼하고 싶다](https://leebaekku.blogspot.com/2023/04/20.html)
* [복권의 당첨확률](lottery.ipynb)
  - 로또6/45의 등수별 당첨 게임수의 신뢰 구간 계산.
  - 번호별 추첨 횟수 데이터를 사용하여 각 번호별 추첨될 확률을 업데이트(베이즈 추정)
  - 관련 포스팅: [준비하시고 쏘세요](https://leebaekku.blogspot.com/2023/04/blog-post.html)
