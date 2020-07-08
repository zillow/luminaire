import pandas.tseries.holiday


class LADHolidays(pandas.tseries.holiday.AbstractHolidayCalendar):
    """
    A class that generates holiday calendars to be used as external features in the batch outlier detection model.
    By default, holidays include:

    - Memorial Day, plus the weekend leading into it
    - Veterans Day, plus the weekend leading into it
    - Labor Day
    - President's Day
    - Martin Luther King Jr. Day
    - Valentine's Day
    - Mother's Day
    - Father's Day
    - Independence Day (actual and observed)
    - Halloween
    - Superbowl
    - Easter
    - Thanksgiving, plus the following weekend
    - Christmas Eve, Christmas Day, and all dates up to New Year's Day (actual and observed)
    """
    from pandas import DateOffset as _DateOffset
    import pandas.tseries.holiday as _H

    USMemorialDay = _H.Holiday('Memorial Day', month=5, day=31,
                            offset=_DateOffset(weekday=_H.MO(-1)))
    USMemorialDaySun = _H.Holiday('Memorial Day Sunday', month=5, day=31,
                               offset=[_DateOffset(weekday=_H.MO(-1)), _DateOffset(days=-1)])
    USMemorialDaySat = _H.Holiday('Memorial Day Saturday', month=5, day=31,
                               offset=[_DateOffset(weekday=_H.MO(-1)), _DateOffset(days=-2)])
    USMemorialDayFri = _H.Holiday('Memorial Day Friday', month=5, day=31,
                               offset=[_DateOffset(weekday=_H.MO(-1)), _DateOffset(days=-3)])

    USVeteransDay = _H.Holiday('Veterans Day', month=11, day=11, observance=_H.nearest_workday)

    USLaborDaySun = _H.Holiday('Labor Day Sunday', month=9, day=1,
                            offset=[_DateOffset(weekday=_H.MO(1)), _DateOffset(days=-1)])
    USLaborDay = _H.Holiday('Labor Day', month=9, day=1,
                            offset=_DateOffset(weekday=_H.MO(1)))
    USLaborDayTues = _H.Holiday('Labor Day Tues', month=9, day=1,
                            offset=[_DateOffset(weekday=_H.MO(1)), _DateOffset(days=1)])

    USPresidentsDay = _H.Holiday('Presidents Day', month=2, day=1, offset=_DateOffset(weekday=_H.MO(3)))
    USMartinLutherKingJr = _H.Holiday('MLK Day', month=1, day=1, offset=_DateOffset(weekday=_H.MO(3)))

    USValentinesDay = _H.Holiday('Valentines Day', month=2, day=14)

    MothersDay = _H.Holiday('Mothers Day', month=5, day=1, offset=_DateOffset(weekday=_H.SU(2)))
    FathersDay = _H.Holiday('Fathers Day', month=6, day=1, offset=_DateOffset(weekday=_H.SU(3)))

    July4th = _H.Holiday('July 4th', month=7, day=4)
    July4thObserved = _H.Holiday('July 4th Observed', month=7, day=4, observance=_H.nearest_workday)

    Halloween = _H.Holiday('Halloween', month=10, day=31)
    Superbowl = _H.Holiday('Superbowl', month=2, day=1,
                            offset=_DateOffset(weekday=_H.SU(1)))
    Easter = _H.Holiday('Easter', month=1, day=1, offset=[_H.Easter(), _H.Day(0)])

    USThanksgivingDay = _H.Holiday('Thanksgiving', month=11, day=1,
                                offset=_DateOffset(weekday=_H.TH(4)))
    USThanksgivingFriday = _H.Holiday('Thanksgiving Friday', month=11, day=1,
                                   offset=[_DateOffset(weekday=_H.TH(4)), _DateOffset(days=1)])
    USThanksgivingSaturday = _H.Holiday('Thanksgiving Saturday', month=11, day=1,
                                     offset=[_DateOffset(weekday=_H.TH(4)), _DateOffset(days=2)])
    USThanksgivingSunday = _H.Holiday('Thanksgiving Sunday', month=11, day=1,
                                   offset=[_DateOffset(weekday=_H.TH(4)), _DateOffset(days=3)])

    ChristmasEve = _H.Holiday('Christmas Eve', month=12, day=24)
    ChristmasDay = _H.Holiday('Christmas Day', month=12, day=25)
    ChristmasDayObserved = _H.Holiday('Christmas Day Observed', month=12, day=25, observance=_H.nearest_workday)
    Dec26 = _H.Holiday('December 26', month=12, day=26)
    Dec27 = _H.Holiday('December 27', month=12, day=27)
    Dec28 = _H.Holiday('December 28', month=12, day=28)
    Dec29 = _H.Holiday('December 29', month=12, day=29)
    Dec30 = _H.Holiday('December 30', month=12, day=30)
    Dec31 = _H.Holiday('New Years Eve', month=12, day=31)

    NewYearsDay = _H.Holiday('New Years Day', month=1, day=1)
    NewYearsDayObserved = _H.Holiday('New Years Day Observed', month=1, day=1, observance=_H.nearest_workday)

    rules = [
        USMemorialDay,
        USMemorialDaySun,
        USMemorialDaySat,
        USMemorialDayFri,

        USVeteransDay,
        USLaborDaySun,
        USLaborDay,
        USLaborDayTues,
        USPresidentsDay,
        USMartinLutherKingJr,

        USValentinesDay,

        MothersDay,
        FathersDay,

        July4th,
        July4thObserved,

        Halloween,
        Superbowl,
        Easter,

        USThanksgivingDay,
        USThanksgivingFriday,
        USThanksgivingSaturday,
        USThanksgivingSunday,
        ChristmasEve,
        ChristmasDay,
        ChristmasDayObserved,
        Dec26,
        Dec27,
        Dec28,
        Dec29,
        Dec30,
        Dec31,
        NewYearsDay,
        NewYearsDayObserved,
    ]

    def __init__(self, name=None, holiday_rules=None):
        if holiday_rules:
            self.__class__.rules = holiday_rules
        super(LADHolidays, self).__init__(name, self.__class__.rules)

