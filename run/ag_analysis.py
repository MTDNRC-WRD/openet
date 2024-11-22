
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import geopandas as gpd
from scipy.stats import linregress

from run_all import COUNTIES
from run_all import MANAGEMENT_FACTORS


def one_county_crop(county):
    # Building data for one county.
    species = (1997, 2002, 2007, 2012, 2017, 2022)
    weight_counts = {}
    for crop in ['HAY & HAYLAGE, IRRIGATED - ACRES HARVESTED', 'WHEAT, IRRIGATED - ACRES HARVESTED',
                 'BARLEY, IRRIGATED - ACRES HARVESTED', 'SUGARBEETS, IRRIGATED - ACRES HARVESTED',
                 'CORN, GRAIN, IRRIGATED - ACRES HARVESTED',
                 'BEANS, DRY EDIBLE, (EXCL CHICKPEAS & LIMA), IRRIGATED - ACRES HARVESTED']:
        one_crop = county_crops[county_crops['Data Item'] == crop]
        vals = np.zeros(6)
        try:
            one_cnty_crop = one_crop.loc[county]
            # print(one_cnty_crop['Value'])
            for i in range(6):
                try:
                    # vals[i] = one_cnty_crop.loc[species[i]]['Value']  # raw acres
                    vals[i] = (one_cnty_crop.loc[species[i]]['Value']
                               / county_sum.at[(county, species[i]), 'tot_irr_acres'])  # fraction of county
                    if crop == 'HAY & HAYLAGE, IRRIGATED - ACRES HARVESTED' and i == 5:
                        print("{} Total: {}, Hay: {}".format(county,
                                                             county_sum.at[(county, species[i]), 'tot_irr_acres'],
                                                             one_cnty_crop.loc[species[i]]['Value']))
                except KeyError:
                    vals[i] = 0
        except KeyError:
            vals = np.zeros(6)
        weight_counts[crop] = vals
    # print(weight_counts)

    width = 1.0

    fig, ax = plt.subplots()
    bottom = np.zeros(6)

    for boolean, weight_count in weight_counts.items():
        ax.bar(species, weight_count, width, label=boolean, bottom=bottom, zorder=3)
        bottom += weight_count

    ax.grid(zorder=0)
    ax.set_ylim(0, 1)
    ax.set_title("{} County ({})".format(COUNTIES[county], county))
    ax.legend()
    ax.set_xlabel('Year')


def one_year_crop_frac(year):
    crops = {'HAY & HAYLAGE': 'HAY & HAYLAGE, IRRIGATED - ACRES HARVESTED',
             'WHEAT': 'WHEAT, IRRIGATED - ACRES HARVESTED',
             'BARLEY': 'BARLEY, IRRIGATED - ACRES HARVESTED',
             'SUGARBEETS': 'SUGARBEETS, IRRIGATED - ACRES HARVESTED',
             'CORN': 'CORN, GRAIN, IRRIGATED - ACRES HARVESTED',
             'BEANS': 'BEANS, DRY EDIBLE, (EXCL CHICKPEAS & LIMA), IRRIGATED - ACRES HARVESTED'}
    species = COUNTIES.keys()
    weight_counts = {}
    for crop in crops.keys():
        vals = []
        if crop != 'CORN':
            one_crop = county_crops[county_crops['Data Item'] == crops[crop]]
            for i in species:
                try:
                    # vals.append(one_crop.at[(i, year), 'Value'])  # raw acres
                    vals.append(one_crop.at[(i, year), 'Value']
                                / county_sum.at[(i, year), 'tot_irr_acres'])  # fraction of county
                    if crop == 'HAY & HAYLAGE':
                        print("{} Total: {}, Hay: {}".format(i,
                                                             county_sum.at[(i, year), 'tot_irr_acres'],
                                                             one_crop.at[(i, year), 'Value']))
                except KeyError:
                    vals.append(0)
        else:
            one_crop = county_crops[county_crops['Data Item'] == 'CORN, GRAIN, IRRIGATED - ACRES HARVESTED']
            other_crop = county_crops[county_crops['Data Item'] == 'CORN, SILAGE, IRRIGATED - ACRES HARVESTED']
            for i in species:
                try:
                    # vals.append(one_crop.at[(i, year), 'Value'] + other_crop.at[(i, year), 'Value'])  # raw acres
                    vals.append((one_crop.at[(i, year), 'Value'] + other_crop.at[(i, year), 'Value'])
                                / county_sum.at[(i, year), 'tot_irr_acres'])  # fraction of county
                except KeyError:
                    vals.append(0)
        weight_counts[crop] = np.array(vals)
    # print(weight_counts)

    width = 0.9

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(56)

    for boolean, weight_count in weight_counts.items():
        ax.bar(species, weight_count, width, label=boolean, bottom=bottom, zorder=3)
        bottom += weight_count

    ax.grid(zorder=0)
    ax.set_ylim(0, 1)
    ax.set_title(year)
    # ax.legend(loc="upper right")
    ax.tick_params(axis='x', rotation=90)
    ax.legend()
    ax.set_xlabel('County')
    ax.set_ylabel('Fraction of Irrigated Acreage')


def one_year_crop_acre(year):
    crops = {'HAY & HAYLAGE': 'HAY & HAYLAGE, IRRIGATED - ACRES HARVESTED',
             'WHEAT': 'WHEAT, IRRIGATED - ACRES HARVESTED',
             'BARLEY': 'BARLEY, IRRIGATED - ACRES HARVESTED',
             'SUGARBEETS': 'SUGARBEETS, IRRIGATED - ACRES HARVESTED',
             'CORN': 'CORN, GRAIN, IRRIGATED - ACRES HARVESTED',
             'BEANS': 'BEANS, DRY EDIBLE, (EXCL CHICKPEAS & LIMA), IRRIGATED - ACRES HARVESTED'}
    species = COUNTIES.keys()
    weight_counts = {}
    for crop in crops.keys():
        # print(crop)
        vals = []
        if crop != 'CORN':
            one_crop = county_crops[county_crops['Data Item'] == crops[crop]]
            # print(one_crop)
            for i in species:
                # print(i)
                try:
                    vals.append(one_crop.at[(i, year), 'Value'])  # raw acres
                    # vals.append(one_crop.at[(i, year), 'Value']
                    #             / county_sum.at[(i, year), 'tot_irr_acres'])  # fraction of county
                    if crop == 'HAY & HAYLAGE':
                        print("{} Total: {}, Hay: {}".format(i,
                                                             county_sum.at[(i, year), 'tot_irr_acres'],
                                                             one_crop.at[(i, year), 'Value']))
                except KeyError:
                    vals.append(0)
        else:
            one_crop = county_crops[county_crops['Data Item'] == 'CORN, GRAIN, IRRIGATED - ACRES HARVESTED']
            other_crop = county_crops[county_crops['Data Item'] == 'CORN, SILAGE, IRRIGATED - ACRES HARVESTED']
            # print(one_crop)
            for i in species:
                # print(i)
                try:
                    # print(i, one_crop.at[(i, year), 'Value'], other_crop.at[(i, year), 'Value'])
                    # vals.append(one_crop.at[(i, year), 'Value'] + other_crop.at[(i, year), 'Value'])  # raw acres
                    vals.append((one_crop.at[(i, year), 'Value'] + other_crop.at[(i, year), 'Value'])
                                / county_sum.at[(i, year), 'tot_irr_acres'])  # fraction of county
                except KeyError:
                    vals.append(0)
        weight_counts[crop] = np.array(vals)
    # print(weight_counts)

    width = 0.9

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(56)

    for boolean, weight_count in weight_counts.items():
        ax.bar(species, weight_count, width, label=boolean, bottom=bottom, zorder=3)
        bottom += weight_count

    ax.grid(zorder=0)
    # ax.set_ylim(0, 1)
    ax.set_title(year)
    # ax.legend(loc="upper right")
    ax.tick_params(axis='x', rotation=90)
    ax.set_xlabel('County')
    ax.set_ylabel('Irrigated Acres Harvested')
    ax.legend()


def one_year_crop_frac_1(year):
    """ Uses SID as total acreage. """

    # getting total acres by county in SID.
    # Load Statewide Irrigation Dataset (about 10 seconds)
    mt_file = "F:/openet_pilot/statewide_irrigation_dataset_15FEB2024_5071.shp"
    mt_fields = gpd.read_file(mt_file)  # takes a bit (8.3s)
    mt_fields['county'] = mt_fields['FID'].str.slice(0, 3)
    mt_fields['area_m2'] = mt_fields['geometry'].area / 4047  # convert to acres

    sid_tot_irr_acres = mt_fields.groupby('county')['area_m2'].sum()
    nass_tot_irr_acres = county_sum.xs(year, level='Year')['tot_irr_acres']
    nass_tot_irr_acres_1 = county_sum.xs(2017, level='Year')['tot_irr_acres']

    sid_tot_irr_acres = sid_tot_irr_acres.sort_index()
    nass_tot_irr_acres = nass_tot_irr_acres.sort_index()
    nass_tot_irr_acres_1 = nass_tot_irr_acres_1.sort_index()
    nass_tot_irr_acres_ind = nass_tot_irr_acres[nass_tot_irr_acres.index.isin(sid_tot_irr_acres.index)]

    # res1 = linregress(nass_tot_irr_acres_ind.to_numpy().astype(float)[1:],
    #                   sid_tot_irr_acres.to_numpy().astype(float)[1:])
    # res2 = linregress(nass_tot_irr_acres_1[nass_tot_irr_acres_1.index.isin(sid_tot_irr_acres.index)]
    #                   .to_numpy().astype(float)[1:],
    #                   sid_tot_irr_acres.to_numpy().astype(float)[1:])

    # plt.figure()  # SID tends to be a little higher
    # plt.scatter(nass_tot_irr_acres_ind, sid_tot_irr_acres, label=year)
    # plt.scatter(nass_tot_irr_acres_1[nass_tot_irr_acres_1.index.isin(sid_tot_irr_acres.index)],
    #             sid_tot_irr_acres, label=2017)
    # plt.plot(nass_tot_irr_acres_ind, res1.intercept + res1.slope*nass_tot_irr_acres_ind,
    #          label='{} m:{:.2f} r^2:{:.2f}'.format(year, res1.slope, res1.rvalue**2))
    # plt.plot(nass_tot_irr_acres_1[nass_tot_irr_acres_1.index.isin(sid_tot_irr_acres.index)],
    #          res2.intercept +
    #          res2.slope*nass_tot_irr_acres_1[nass_tot_irr_acres_1.index.isin(sid_tot_irr_acres.index)],
    #          label='2017 m:{:.2f} r^2:{:.2f}'.format(res2.slope, res2.rvalue**2))
    # plt.plot(sid_tot_irr_acres, sid_tot_irr_acres)
    # plt.xlabel('NASS Irrigated Acreage')
    # plt.ylabel('SID Irrigated Acreage')
    # plt.legend()
    # plt.grid()

    # for i in range(53):  # 001 Beaverhead, NASS indicates twice the irrigated acreage
    #     print("NASS: {} {:.1f} SID: {} {}".format(nass_tot_irr_acres_ind.index[i], nass_tot_irr_acres_ind.iloc[i],
    #                                               sid_tot_irr_acres.index[i], sid_tot_irr_acres.iloc[i]))

    # Actual stuff

    crops = {'HAY & HAYLAGE': 'HAY & HAYLAGE, IRRIGATED - ACRES HARVESTED',
             'WHEAT': 'WHEAT, IRRIGATED - ACRES HARVESTED',
             'BARLEY': 'BARLEY, IRRIGATED - ACRES HARVESTED',
             'SUGARBEETS': 'SUGARBEETS, IRRIGATED - ACRES HARVESTED',
             'CORN': 'CORN, GRAIN, IRRIGATED - ACRES HARVESTED',
             'BEANS': 'BEANS, DRY EDIBLE, (EXCL CHICKPEAS & LIMA), IRRIGATED - ACRES HARVESTED'}
    species = COUNTIES.keys()
    weight_counts = {}
    for crop in crops.keys():
        # print(crop)
        vals = []
        if crop != 'CORN':
            one_crop = county_crops[county_crops['Data Item'] == crops[crop]]
            for i in species:
                try:
                    # vals.append(one_crop.at[(i, year), 'Value'])  # raw acres
                    vals.append(one_crop.at[(i, year), 'Value']
                                / sid_tot_irr_acres[i])  # fraction of county
                    # if crop == 'HAY & HAYLAGE':
                    #     print("{} Total: {}, SID Total {:.1f}, Hay: {}"
                    #           .format(i, county_sum.at[(i, year), 'tot_irr_acres'],
                    #                   sid_tot_irr_acres[i], one_crop.at[(i, year), 'Value']))
                except KeyError:
                    vals.append(0)
        else:
            one_crop = county_crops[county_crops['Data Item'] == 'CORN, GRAIN, IRRIGATED - ACRES HARVESTED']
            other_crop = county_crops[county_crops['Data Item'] == 'CORN, SILAGE, IRRIGATED - ACRES HARVESTED']
            for i in species:
                try:
                    # vals.append(one_crop.at[(i, year), 'Value'] + other_crop.at[(i, year), 'Value'])  # raw acres
                    vals.append((one_crop.at[(i, year), 'Value'] + other_crop.at[(i, year), 'Value'])
                                / sid_tot_irr_acres[i])  # fraction of county
                except KeyError:
                    vals.append(0)
        weight_counts[crop] = np.array(vals)
    # print(weight_counts)

    width = 0.9

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(56)

    for boolean, weight_count in weight_counts.items():
        ax.bar(species, weight_count, width, label=boolean, bottom=bottom, zorder=3)
        bottom += weight_count

    ax.grid(zorder=0)
    ax.set_ylim(0, 1)
    ax.set_title(year)
    ax.tick_params(axis='x', rotation=90)
    # ax.legend(title='Values <1 Unknown/No Data')
    ax.legend()
    ax.set_xlabel('County')
    ax.set_ylabel('Fraction of Irrigated Acreage')


def total_irr_acres():
    years = [1997, 2002, 2007, 2012, 2017, 2022]
    species = COUNTIES.keys()
    penguin_means = {}
    print(county_sum.loc['001'].loc[2022]['tot_irr_acres'])
    for i in species:
        for j in years:
            if i == '001':
                penguin_means[j] = [county_sum.loc[i].loc[j]['tot_irr_acres']]
            else:
                penguin_means[j].append(county_sum.loc[i].loc[j]['tot_irr_acres'])

    print(penguin_means)

    x = np.arange(len(species))  # + 1/7  # the label locations
    width = 1/7  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(15, 5))

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        ax.bar(x + offset, measurement, width, label=attribute, zorder=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x + width, species)
    ax.grid(zorder=0)
    ax.tick_params(axis='x', rotation=90)
    ax.legend()
    ax.set_xlabel('County')
    ax.set_ylabel('Irrigated Acreage')


def all_counties_years_crops_acre():
    crops = {'HAY & HAYLAGE': 'HAY & HAYLAGE, IRRIGATED - ACRES HARVESTED',
             'WHEAT': 'WHEAT, IRRIGATED - ACRES HARVESTED',
             'BARLEY': 'BARLEY, IRRIGATED - ACRES HARVESTED',
             'SUGARBEETS': 'SUGARBEETS, IRRIGATED - ACRES HARVESTED',
             'CORN': 'CORN, GRAIN, IRRIGATED - ACRES HARVESTED',
             'BEANS': 'BEANS, DRY EDIBLE, (EXCL CHICKPEAS & LIMA), IRRIGATED - ACRES HARVESTED'}

    years = [1997, 2002, 2007, 2012, 2017, 2022]
    species = COUNTIES.keys()
    penguin_means = {}
    for j in years:
        weight_counts = {}
        for crop in crops.keys():
            # print(crop)
            vals = []
            if crop != 'CORN':
                one_crop = county_crops[county_crops['Data Item'] == crops[crop]]
                for i in species:
                    try:
                        vals.append(one_crop.at[(i, j), 'Value'])  # raw acres
                        # vals.append(one_crop.at[(i, year), 'Value']
                        #             / county_sum.at[(i, year), 'tot_irr_acres'])  # fraction of county
                        if crop == 'HAY & HAYLAGE':
                            print("{} Total: {}, Hay: {}".format(i,
                                                                 county_sum.at[(i, j), 'tot_irr_acres'],
                                                                 one_crop.at[(i, j), 'Value']))
                    except KeyError:
                        vals.append(0)
            else:
                one_crop = county_crops[county_crops['Data Item'] == 'CORN, GRAIN, IRRIGATED - ACRES HARVESTED']
                other_crop = county_crops[county_crops['Data Item'] == 'CORN, SILAGE, IRRIGATED - ACRES HARVESTED']
                for i in species:
                    try:
                        # print(i, one_crop.at[(i, j), 'Value'], other_crop.at[(i, j), 'Value'])
                        vals.append(one_crop.at[(i, j), 'Value'] + other_crop.at[(i, j), 'Value'])  # raw acres
                        # vals.append((one_crop.at[(i, j), 'Value'] + other_crop.at[(i, j), 'Value'])
                        #             / county_sum.at[(i, j), 'tot_irr_acres'])  # fraction of county
                    except KeyError:
                        vals.append(0)
            weight_counts[crop] = np.array(vals)
        penguin_means[j] = weight_counts

    x = np.arange(len(species))  # + 1/7  # the label locations
    width = 1/7  # the width of the bars
    multiplier = 0

    clrs = dict(zip(crops.keys(), mpl.colormaps['tab10'].colors[:6]))

    fig, ax = plt.subplots(layout='constrained', figsize=(15, 5))
    first = 0
    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        bottom = np.zeros(56)
        if first == 0:
            for boolean, weight_count in measurement.items():
                ax.bar(x + offset, weight_count, width, label=boolean, bottom=bottom, zorder=3, color=clrs[boolean])
                bottom += weight_count
            first = 1
        else:
            for boolean, weight_count in measurement.items():
                ax.bar(x + offset, weight_count, width, bottom=bottom, zorder=3, color=clrs[boolean])
                bottom += weight_count
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x + width, species)
    ax.grid(zorder=0)
    ax.tick_params(axis='x', rotation=90)
    ax.legend()
    ax.set_xlabel('County')
    ax.set_ylabel('Irrigated Acres Harvested')


def all_counties_years_crops_frac():
    crops = {'HAY & HAYLAGE': 'HAY & HAYLAGE, IRRIGATED - ACRES HARVESTED',
             'WHEAT': 'WHEAT, IRRIGATED - ACRES HARVESTED',
             'BARLEY': 'BARLEY, IRRIGATED - ACRES HARVESTED',
             'SUGARBEETS': 'SUGARBEETS, IRRIGATED - ACRES HARVESTED',
             'CORN': 'CORN, GRAIN, IRRIGATED - ACRES HARVESTED',
             'BEANS': 'BEANS, DRY EDIBLE, (EXCL CHICKPEAS & LIMA), IRRIGATED - ACRES HARVESTED'}

    years = [1997, 2002, 2007, 2012, 2017, 2022]
    species = COUNTIES.keys()
    penguin_means = {}
    for j in years:
        weight_counts = {}
        for crop in crops.keys():
            # print(crop)
            vals = []
            if crop != 'CORN':
                one_crop = county_crops[county_crops['Data Item'] == crops[crop]]
                for i in species:
                    try:
                        # vals.append(one_crop.at[(i, j), 'Value'])  # raw acres
                        vals.append(one_crop.at[(i, j), 'Value']
                                    / county_sum.at[(i, j), 'tot_irr_acres'])  # fraction of county
                        if crop == 'HAY & HAYLAGE':
                            print("{} Total: {}, Hay: {}".format(i,
                                                                 county_sum.at[(i, j), 'tot_irr_acres'],
                                                                 one_crop.at[(i, j), 'Value']))
                    except KeyError:
                        vals.append(0)
            else:
                one_crop = county_crops[county_crops['Data Item'] == 'CORN, GRAIN, IRRIGATED - ACRES HARVESTED']
                other_crop = county_crops[county_crops['Data Item'] == 'CORN, SILAGE, IRRIGATED - ACRES HARVESTED']
                for i in species:
                    try:
                        # print(i, one_crop.at[(i, j), 'Value'], other_crop.at[(i, j), 'Value'])
                        # vals.append(one_crop.at[(i, j), 'Value'] + other_crop.at[(i, j), 'Value'])  # raw acres
                        vals.append((one_crop.at[(i, j), 'Value'] + other_crop.at[(i, j), 'Value'])
                                    / county_sum.at[(i, j), 'tot_irr_acres'])  # fraction of county
                    except KeyError:
                        vals.append(0)
            weight_counts[crop] = np.array(vals)
        penguin_means[j] = weight_counts

    x = np.arange(len(species))  # + 1/7  # the label locations
    width = 1/7  # the width of the bars
    multiplier = 0

    clrs = dict(zip(crops.keys(), mpl.colormaps['tab10'].colors[:6]))

    fig, ax = plt.subplots(layout='constrained', figsize=(15, 5))
    first = 0
    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        bottom = np.zeros(56)
        if first == 0:
            for boolean, weight_count in measurement.items():
                ax.bar(x + offset, weight_count, width, label=boolean, bottom=bottom, zorder=3, color=clrs[boolean])
                bottom += weight_count
            first = 1
        else:
            for boolean, weight_count in measurement.items():
                ax.bar(x + offset, weight_count, width, bottom=bottom, zorder=3, color=clrs[boolean])
                bottom += weight_count
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x + width, species)
    ax.grid(zorder=0)
    ax.tick_params(axis='x', rotation=90)
    ax.legend()
    ax.set_xlabel('County')
    ax.set_ylabel('Fraction of Irrigated Acreage')


def all_counties_years_crops_both(scatter=False):
    # Initialization
    crops = {'HAY & HAYLAGE': 'HAY & HAYLAGE, IRRIGATED - ACRES HARVESTED',
             'WHEAT': 'WHEAT, IRRIGATED - ACRES HARVESTED',
             'BARLEY': 'BARLEY, IRRIGATED - ACRES HARVESTED',
             'SUGARBEETS': 'SUGARBEETS, IRRIGATED - ACRES HARVESTED',
             'CORN': 'CORN, GRAIN, IRRIGATED - ACRES HARVESTED',
             'BEANS': 'BEANS, DRY EDIBLE, (EXCL CHICKPEAS & LIMA), IRRIGATED - ACRES HARVESTED'}
    years = [1997, 2002, 2007, 2012, 2017, 2022]
    species = COUNTIES.keys()

    # General plotting information
    fig, axs = plt.subplots(nrows=2, layout='constrained', figsize=(18, 10))
    plt.suptitle('Montana Irrigated Agriculture (NASS, 1997-2022)')
    x = np.arange(len(species))  # + 1/7  # the label locations
    width = 1 / 7  # the width of the bars
    multiplier = 0
    clrs = dict(zip(crops.keys(), mpl.colormaps['tab10'].colors[:6]))

    # Setting up data
    penguin_means1 = {}
    penguin_means2 = {}
    for j in years:
        weight_counts1 = {}
        weight_counts2 = {}
        for crop in crops.keys():
            # print(crop)
            vals1 = []
            vals2 = []
            if crop != 'CORN':
                one_crop = county_crops[county_crops['Data Item'] == crops[crop]]
                for i in species:
                    try:
                        vals2.append(one_crop.at[(i, j), 'Value'])  # raw acres
                        vals1.append(one_crop.at[(i, j), 'Value']
                                     / county_sum.at[(i, j), 'tot_irr_acres'])  # fraction of county
                        # if crop == 'HAY & HAYLAGE':
                        #     print("{} Total: {}, Hay: {}".format(i,
                        #                                          county_sum.at[(i, j), 'tot_irr_acres'],
                        #                                          one_crop.at[(i, j), 'Value']))
                    except KeyError:
                        vals1.append(0)
                        vals2.append(0)
            else:
                one_crop = county_crops[county_crops['Data Item'] == 'CORN, GRAIN, IRRIGATED - ACRES HARVESTED']
                other_crop = county_crops[county_crops['Data Item'] == 'CORN, SILAGE, IRRIGATED - ACRES HARVESTED']
                for i in species:
                    try:
                        # print(i, one_crop.at[(i, j), 'Value'], other_crop.at[(i, j), 'Value'])
                        vals2.append(one_crop.at[(i, j), 'Value'] + other_crop.at[(i, j), 'Value'])  # raw acres
                        vals1.append((one_crop.at[(i, j), 'Value'] + other_crop.at[(i, j), 'Value'])
                                     / county_sum.at[(i, j), 'tot_irr_acres'])  # fraction of county
                    except KeyError:
                        vals1.append(0)
                        vals2.append(0)
            weight_counts1[crop] = np.array(vals1)
            weight_counts2[crop] = np.array(vals2)
        penguin_means1[j] = weight_counts1
        penguin_means2[j] = weight_counts2

    # First subplot
    first = 0
    for attribute, measurement in penguin_means2.items():
        offset = width * multiplier
        bottom = np.zeros(56)
        if first == 0:
            for boolean, weight_count in measurement.items():
                axs[0].bar(x + offset, weight_count, width, label=boolean,
                           bottom=bottom, zorder=3, color=clrs[boolean])
                bottom += weight_count
            first = 1
        else:
            for boolean, weight_count in measurement.items():
                axs[0].bar(x + offset, weight_count, width, bottom=bottom, zorder=3, color=clrs[boolean])
                bottom += weight_count
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[0].set_xticks(x + width, species)
    axs[0].grid(zorder=0)
    axs[0].tick_params(axis='x', rotation=90)
    axs[0].legend(title='Crops by Statewide Prevalence')
    axs[0].legend()
    # axs[0].set_xlabel('County')
    axs[0].set_ylabel('Irrigated Acres Harvested')

    # Second subplot
    multiplier = 0
    per_not_hay_by_year = np.zeros((6, 56))
    err_by_year = np.zeros((5, 56))
    for attribute, measurement in penguin_means1.items():
        offset = width * multiplier
        bottom = np.zeros(56)
        for boolean, weight_count in measurement.items():
            axs[1].bar(x + offset, weight_count, width, bottom=bottom, zorder=3, color=clrs[boolean])
            if boolean == 'HAY & HAYLAGE':
                temp = weight_count
            bottom += weight_count
        per_not_hay_by_year[multiplier] = bottom - temp
        if attribute != 1997:
            err_by_year[multiplier - 1] = 1.0 - bottom
        multiplier += 1
    per_not_hay = np.nanmean(per_not_hay_by_year, axis=0)
    err = np.nanmean(err_by_year, axis=0)
    # print(per_not_hay)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[1].set_xticks(x + width, species)
    axs[1].grid(zorder=0)
    axs[1].tick_params(axis='x', rotation=90)
    # axs[1].legend()
    axs[1].set_xlabel('County')
    axs[1].set_ylabel('Fraction of Irrigated Acreage')

    if scatter:
        # Plot the relationship between percentage of non-hay crops and difference in DNRC vs OpenET CU estimates
        dif = [4.178644649288044, 10.640930038327411, 2.4918067681642686, 7.073371224783596, 4.307162415640173,
               5.471854792633717, 7.131116372653963, 9.490607253597194, 8.286047278532587, 7.4232688641597075,
               7.893001127672255, 2.130801450160476, 8.96632413290408, 8.531114743667231, 1.353267170138956,
               3.5197686412467153, 4.5909783653265634, 4.701661904841291, 1.034487105853584, 4.134805663833527,
               2.876529316070659, 3.299805104442669, 5.817191037324497, 2.644913835095336, 2.779981647285352,
               5.7725873698072405, 3.785305612540533, 3.406655343974659, 4.162166998922876, 1.9971765519938245,
               1.8545806536421026, 2.5506093941355115, 0.9282433079787804, 1.124720277879943, 9.26493482732734,
               2.9855693048802614, 6.449997005111632, 10.356357844070418, 5.657526576774178, 10.041382140725583,
               8.205520893330906, 7.040680316541794, 2.448590753784197, 9.735281229038566, 5.631787451381328,
               4.116770643626053, 1.4295042105767468, 8.13170946030604, 12.075041199358855, 7.172158791074054,
               0.5238398098584796, 6.547394174654958]
        mf = []
        for key, i in MANAGEMENT_FACTORS.items():
            # print(i)
            mf.append(i[2])
        mf = np.asarray(mf)

        res_drop = [per_not_hay[:5], per_not_hay[6:12], per_not_hay[13:50],
                    per_not_hay[51:54], [per_not_hay[55]]]
        res_drop = np.concatenate(res_drop).ravel()
        err_drop = [err[:5], err[6:12], err[13:50], err[51:54], [err[55]]]
        err_drop = np.concatenate(err_drop).ravel()
        err_drop = np.nan_to_num(err_drop, posinf=0.0, neginf=0.99)
        mf_drop = [mf[:5], mf[6:12], mf[13:50], mf[51:54], [mf[55]]]
        mf_drop = np.concatenate(mf_drop).ravel()

        res_drop1 = np.nan_to_num(res_drop, posinf=0.0)
        res1 = linregress(res_drop1, dif)
        res2 = linregress(res_drop1, mf_drop)
        res3 = linregress(mf_drop, dif)

        plt.figure()
        # plt.scatter(res_drop, dif, zorder=2)
        for i in range(len(dif)):
            plt.errorbar(res_drop[i], dif[i], xerr=err_drop[i], zorder=2, fmt='o', alpha=(1-err_drop[i])**2,
                         color='tab:green', mec='none')
        # plt.errorbar(res_drop, dif, xerr=err_drop, zorder=2, fmt='o')
        plt.plot(res_drop, res1.intercept + res1.slope*res_drop, zorder=4,
                 label='r^2: {:.2f}'.format(res1.rvalue**2), color='tab:purple')
        plt.grid(zorder=1)
        plt.xlabel("Average fraction of non-hay crops")
        plt.ylabel("Average difference in CU (DNRC minus OpenET)")
        plt.xlim(0, 1)
        plt.legend()

        # # Adding stuff with the management factor. Not all that interesting.
        # # Difference in CU estimate might be mildly related to management factor, though mf and percent of non-hay
        # # crop do not appear to be related.
        # plt.figure(figsize=(10, 5))
        #
        # plt.subplot(121)
        # # plt.scatter(res_drop, mf_drop, zorder=2)
        # for i in range(len(mf_drop)):
        #     plt.errorbar(res_drop[i], mf_drop[i], xerr=err_drop[i], zorder=2, fmt='o', alpha=(1 - err_drop[i]) ** 2,
        #                  color='tab:green', mec='none')
        # # plt.errorbar(res_drop, mf, xerr=err_drop, zorder=2, fmt='o')
        # plt.plot(res_drop, res2.intercept + res2.slope * res_drop, zorder=4,
        #          label='r^2: {:.2f}'.format(res2.rvalue ** 2), color='tab:purple')
        # plt.grid(zorder=1)
        # plt.xlabel("Average fraction of non-hay crops")
        # plt.ylabel("Value of 1997-2006 management factor")
        # plt.xlim(0, 1)
        # plt.legend()
        #
        # plt.subplot(122)
        # plt.scatter(mf_drop, dif, zorder=2)
        # # for i in range(len(mf_drop)):
        # #     plt.errorbar(res_drop[i], mf_drop[i], xerr=err_drop[i], zorder=2, fmt='o', alpha=(1 - err_drop[i]) ** 2,
        # #                  color='tab:green', mec='none')
        # # plt.errorbar(res_drop, mf, xerr=err_drop, zorder=2, fmt='o')
        # plt.plot(mf_drop, res3.intercept + res3.slope * mf_drop, zorder=4,
        #          label='r^2: {:.2f}'.format(res3.rvalue ** 2), color='tab:purple')
        # plt.grid(zorder=1)
        # plt.ylabel("Average difference in CU (DNRC minus OpenET)")
        # plt.xlabel("Value of 1997-2006 management factor")
        # # plt.xlim(0, 1)
        # plt.legend()

        # # Plotting histogram of average unknown crop percentage by county
        # plt.figure()
        # print(np.linspace(0, 1, 20) - 0.025)
        # plt.hist(err_drop, bins=np.linspace(0, 1, 20)-0.025, zorder=2, rwidth=0.95)
        # plt.grid(zorder=1)


if __name__ == '__main__':

    # county_totals = pd.read_csv('C:/Users/CND571/Documents/Data/CountyAgCensusData_totalirrigated_1997_2022.csv',
    #                             index_col=['County ANSI', 'Year', 'Data Item'], dtype={'County ANSI': str})
    # county_crops = pd.read_csv('C:/Users/CND571/Documents/Data/CountyAgCensusData_1997_2022.csv',
    #                            index_col=['County ANSI', 'Year', 'Data Item'], dtype={'County ANSI': str})
    county_totals = pd.read_csv('C:/Users/CND571/Documents/Data/CountyAgCensusData_totalirrigated_1997_2022.csv',
                                index_col=['County ANSI', 'Year'], dtype={'County ANSI': str})
    county_crops = pd.read_csv('C:/Users/CND571/Documents/Data/CountyAgCensusData_1997_2022.csv',
                               index_col=['County ANSI', 'Year'], dtype={'County ANSI': str})
    county_totals['Value'] = [i.replace(',', '') for i in county_totals['Value']]
    county_totals['Value'] = pd.to_numeric(county_totals['Value'], errors='coerce')
    county_crops['Value'] = [i.replace(',', '') for i in county_crops['Value']]
    county_crops['Value'] = pd.to_numeric(county_crops['Value'], errors='coerce')
    # print(county_totals.head())
    # print(county_crops.head())

    # print(county_totals.loc['007'].loc[2022]['Value'].sum())

    county_sum = pd.DataFrame(index=county_totals.index.unique(), columns=['tot_irr_acres'])
    for i in county_totals.index.unique():
        des_val = county_totals.loc[i[0]].loc[i[1]]['Value'].sum()
        county_sum.at[(i[0], i[1]), 'tot_irr_acres'] = des_val

    # print(county_sum)

    # NEW THING

    # print(county_crops[county_crops['Data Item']])
    # for i in ['001', '003', '007']:
    #     for j in county_crops['Year'].unique():
    #         print(i, j, county_crops.at[(i, j, 'Value')])

    # one_county_crop('001')
    # one_county_crop('003')
    # one_year_crop_frac(2017)
    one_year_crop_acre(2017)
    # one_year_crop_frac_1(2017)

    # total_irr_acres()

    # all_counties_years_crops_acre()
    # all_counties_years_crops_frac()
    # all_counties_years_crops_both(scatter=True)

    plt.show()

# ========================= EOF ====================================================================
