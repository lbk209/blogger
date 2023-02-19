#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

### trading commission
COMMISSION = 0.0015
SEED_MONEY = 1e7

###
DATE_FORMAT = '%Y-%m-%d'
CLOSE_HOUR = 17
get_today = lambda today: ((today.hour < CLOSE_HOUR) and today - timedelta(days=1)) or today
TODAY_BY_CLOSE = get_today(datetime.today()).strftime(DATE_FORMAT)

