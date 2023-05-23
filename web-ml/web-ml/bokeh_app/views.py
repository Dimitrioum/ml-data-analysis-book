from django.shortcuts import render
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components

import pandas as pd
import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.io import output_file, show, output_notebook, save
from bokeh.models import CustomJS
from bokeh.models.widgets import CheckboxGroup
from bokeh.layouts import row, column
from bokeh.palettes import Category20
from bokeh.models.annotations import Title, Legend
from bokeh.models import LinearAxis, Range1d
Category10 = Category20[20]
from math import sqrt

# df_vis['Время'] = pd.to_datetime(df_vis['Время'])

# df = df_vis

# df['Время'] = pd.to_datetime(df['Время'], format='%d/%m/%Y')

# p1 = figure(x_axis_type='datetime', plot_width=2500)
# p1.extra_y_ranges = {"binary": Range1d(start=-1.5, end=1.5)}
# aline = p1.circle(df['Время'], df['Технологический ящик слева Крышка '], line_width=2, color=Category10[0],
#                  y_range_name="binary")
# bline = p1.circle(df['Время'], df['Технологический ящик справа Крышка '], line_width=2, color=Category10[1],
#                  y_range_name="binary")
# cline = p1.circle(df['Время'], df['Крышку ящика слева открыли'], line_width=2, color=Category10[2],
#                  y_range_name="binary")
# ccline = p1.circle(df['Время'], df['Крышку ящика слева закрыли'], line_width=2, color=Category10[12],
#                  y_range_name="binary")
# dline = p1.circle(df['Время'], df['Крышку ящика справа открыли'], line_width=2, color=Category10[3],
#                  y_range_name="binary")
# ddline = p1.circle(df['Время'], df['Крышку ящика справа закрыли'], line_width=2, color=Category10[4],
#                  y_range_name="binary")
# eline = p1.circle(df['Время'], df['Секция №2 Датчик в сливной магистрали'], line_width=2, color=Category10[5],
#                  y_range_name="binary")
# fline = p1.circle(df['Время'], df['Секция №2 Нефтепродукт пропал из сливной магистрали'], line_width=2, color=Category10[6],
#                  y_range_name="binary")
# gline = p1.circle(df['Время'], df['Cекция №2 Заливная горловина'], line_width=2, color=Category10[13],
#                  y_range_name="binary")
# ggline = p1.circle(df['Время'], df['Отсек №2 Заливную горловину открыли'], line_width=2, color=Category10[15],
#                  y_range_name="binary")
# hline = p1.circle(df['Время'], df['Отсек №2 Заливную горловину закрыли'], line_width=2, color=Category10[7],
#                  y_range_name="binary")
# iline = p1.circle(df['Время'], df['Секция №2 Нефтепродукт появился в сливной магистрали'], line_width=2, color=Category10[8],
#                  y_range_name="binary")
# jline = p1.circle(df['Время'], df['Секция №2 Датчик на дне отсека'], line_width=2, color=Category10[9],
#                  y_range_name="binary")
# kline = p1.circle(df['Время'], df['Секция №2 Нефтепродукт появился на дне отсека'], line_width=2, color=Category10[10],
#                  y_range_name="binary")
# kkline = p1.circle(df['Время'], df['Секция №2 Датчик на уровне планки'], line_width=2, color=Category10[11],
#                  y_range_name="binary")
# lline = p1.circle(df['Время'], df['Скорость'], line_width=2, color=Category10[12])
# mline = p1.circle(df['Время'], df['Секция №2 Уровень нефтепродукта вырос до планки'], line_width=2, color=Category10[18],
#                  y_range_name="binary")
    
# p2 = figure(x_axis_type='datetime', plot_width=10000)
# eline = p1.circle(df['время прихода точки на сервере'], df['Скорость'], line_width=2, color=Viridis6[5])

# p1.yaxis.axis_label = 'Открытие/Закрытие/Событие'
# p1.xaxis.axis_label = 'Время'
# # p2.yaxis.axis_label = 'Скорость'
# # p2.xaxis.axis_label = 'время формирования точки на БВ'

# legend = Legend(items=[
#     ("Секция №1 Заливная горловина", [aline]),
#     ("Секция №1 Датчик на дне отсека", [bline]),
#     ("Секция №1 Датчик в сливной магистрали", [cline]),
#     ("Секция №1 Датчик на уровне планки", [ccline]),
#     ("Секция №1 Уровень НП", [dline]),
#     ("Секция №2 Датчик на уровне планки", [ddline]),
#     ("Cекция №3 Заливная горловина", [eline]),
#     ("Секция №3 Датчик на дне отсека", [fline]),
#     ("Секция №3 Датчик в сливной магистрали", [gline]),
#     ("Секция №3 Датчик на уровне планки", [ggline]),
#     ("Секция №3 Уровень НП", [hline]),
#     ("Cекция №4 Заливная горловина", [iline]),
#     ("Секция №4 Датчик на дне отсека", [jline]),
#     ("Секция №4 Датчик в сливной магистрали", [kline]),
#     ("Секция №4 Датчик на уровне планки", [kkline]),
#     ("Скорость", [lline]),
#     ('Секция №2 Уровень нефтепродукта вырос до планки', [mline]),
# ], location=(0, 250))

# t = Title()
# t.text = 'report_visual'
# p1.title = t
# # p2.title = t
# p1.add_layout(legend, 'left')
# p1.add_layout(LinearAxis(y_range_name="binary"), 'right')
# # p2.add_layout(legend, 'left')
# checkboxes = CheckboxGroup(labels=list(['Технологический ящик слева Крышка ',
#                                        'Технологический ящик справа Крышка ', 'Крышку ящика слева открыли',
#                                        'Крышку ящика слева закрыли', 'Крышку ящика справа открыли',
#                                        'Крышку ящика справа закрыли', 'Секция №2 Датчик в сливной магистрали',
#                                        'Секция №2 Нефтепродукт пропал из сливной магистрали',
#                                        'Cекция №2 Заливная горловина', 'Отсек №2 Заливную горловину открыли',
#                                        'Отсек №2 Заливную горловину закрыли',
#                                        'Секция №2 Нефтепродукт появился в сливной магистрали',
#                                        'Секция №2 Датчик на дне отсека',
#                                        'Секция №2 Нефтепродукт появился на дне отсека',
#                                        'Секция №2 Датчик на уровне планки',
#                                        'Скорость',
#                                        'Секция №2 Уровень нефтепродукта вырос до планки']),
#                            active=[])
# callback = CustomJS(code="""aline.visible = false; // aline and etc.. are 
#                             bline.visible = false; // passed in from args
#                             cline.visible = false;
#                             ccline.visible = false;
#                             dline.visible = false;
#                             ddline.visible = false;
#                             eline.visible = false;
#                             fline.visible = false; 
#                             gline.visible = false;
#                             ggline.visible = false;
#                             hline.visible = false;
#                             iline.visible = false; 
#                             jline.visible = false;
#                             kline.visible = false;
#                             kkline.visible = false;
#                             lline.visible = false;
#                             mline.visible = false;
#                             // cb_obj is injected in thanks to the callback
#                             if (cb_obj.active.includes(0)){aline.visible = true;} 
#                                 // 0 index box is aline
#                             if (cb_obj.active.includes(1)){bline.visible = true;} 
#                                 // 1 index box is bline
#                             if (cb_obj.active.includes(2)){cline.visible = true;} 
#                              if (cb_obj.active.includes(3)){ccline.visible = true;} 
#                                 // 1 index box is bline
#                             if (cb_obj.active.includes(4)){dline.visible = true;} 
#                             if (cb_obj.active.includes(5)){ddline.visible = true;} 
#                                 // 1 index box is bline
#                             if (cb_obj.active.includes(6)){eline.visible = true;} 
#                                 // 1 index box is bline
#                             if (cb_obj.active.includes(7)){fline.visible = true;} 
#                                 // 1 index box is bline
#                             if (cb_obj.active.includes(8)){gline.visible = true;} 
#                              if (cb_obj.active.includes(9)){ggline.visible = true;} 
#                                 // 1 index box is bline
#                             if (cb_obj.active.includes(10)){hline.visible = true;} 
#                                 // 1 index box is bline
#                             if (cb_obj.active.includes(11)){iline.visible = true;} 
#                                 // 1 index box is bline
#                             if (cb_obj.active.includes(12)){jline.visible = true;} 
#                                 // 1 index box is bline
#                             if (cb_obj.active.includes(13)){kline.visible = true;} 
#                             if (cb_obj.active.includes(14)){kkline.visible = true;}
#                             if (cb_obj.active.includes(15)){lline.visible = true;} 
#                             if (cb_obj.active.includes(16)){mline.visible = true;} 
#                                 // 1 index box is bline
#                             """,
#                             args={'aline': aline, 'bline': bline, 'cline': cline, 'ccline': ccline,
#                                   'ddline': ddline, 
#                                   'eline': eline, 'dline': dline, 'fline': fline, 'gline': gline, 'ggline': ggline,
#                                   'hline': hline, 'iline': iline, 'jline': jline, 'kline': kline,
#                                    'kkline': kkline, 'lline': lline, 'mline': mline})
# checkboxes.js_on_click(callback)
# layout = row(p1, checkboxes)
# output_file('report_visual.html')
# # show(column(p1, p2, checkboxes))
# curdoc().add_root(layout)
# curdoc().title="report_visual"

# show(layout)

# Create your views here.
def plot_page(request):
	if "GET" == request.method:
		return render(request, 'plot_page.html', {})
	else:
		excel_file = request.FILES["excel_file"]
		# df = pd.read_excel(excel_file, index=None)
		df = pd.read_excel(excel_file)
		df.columns = df.iloc[3]
		BV_NAME = df.iloc[4][0]
		df = df[5:1000]
		df['Время'] = pd.to_datetime(df['Время'], format='%Y/%m/%d')
		df = df.replace(['сухой', 'мокрый'], [0, 1])
		df = df.replace(['закрыта', 'открыта'], [0, 1])
		# !!!
		df = df.replace(1, 0)
		df['Секция №1 Датчик на дне отсека'].iat[0] = 1
		# !!!
		print(df.head())



        # # getting all sheets
  

        # excel_data = list()
        # # iterating over the rows and
        # # getting value from each cell in row
        # for row in worksheet.iter_rows():
        #     row_data = list()
        #     for cell in row:
        #         row_data.append(str(cell.value))
        #         print(cell.value)
        #     excel_data.append(row_data)


	# y1 = [x * 2 + 5 for x in x]
	# y2 = [x * 3 for x in x]
	# y3 = [sqrt(x) for x in x]
	x = df['Время']
	y_speed = df['Скорость']
	y1 = df['Секция №1 Датчик на дне отсека']
	y2 = df['Секция №1 Датчик в сливной магистрали']
	y3 = df['Секция №1 Датчик на уровне планки']
	y4 = df['Cекция №1 Заливная горловина']
	y5 = df['Секция №2 Датчик на дне отсека']
	y6 = df['Секция №2 Датчик на уровне планки']
	y7 = df['Секция №2 Датчик в сливной магистрали']
	y8 = df['Cекция №2 Заливная горловина']
	y9 = df['Секция №3 Датчик на дне отсека']
	y10 = df['Секция №3 Датчик на уровне планки']
	y11 = df['Секция №3 Датчик в сливной магистрали']
	y12 = df['Cекция №3 Заливная горловина']
	y13 = df['Секция №4 Датчик на дне отсека']
	y14 = df['Секция №4 Датчик на уровне планки']
	y15 = df['Секция №4 Датчик в сливной магистрали']
	y16 = df['Cекция №4 Заливная горловина']
	y17 = df['Технологический ящик слева Крышка ']
	y18 = df['Технологический ящик справа Крышка ']

	

	plot = figure(title='ОТЧЕТ' + ' ' + BV_NAME, x_axis_label='Время (месяц/день)', y_axis_label='Скорость, км/ч', sizing_mode='stretch_both', x_axis_type='datetime')
	plot.extra_y_ranges = {"binary": Range1d(start=-0.5, end=1.5)}
	plot.add_layout(LinearAxis(y_range_name="binary", axis_label='1 - Открыт(Мокрый)/ 0 - Закрыт(Сухой)'), 'right')

	# y1_line = plot.circle(x, y1, line_width=2, color=Category10[6])
	# y2_line = plot.circle(x, y2, line_width=2, color=Category10[1])
	# y3_line = plot.circle(x, y3, line_width=2, color=Category10[2])
	y_speed_line = plot.circle(x, y_speed, line_width=2, color=Category10[0])
	y1_line = plot.circle(x, y1, line_width=2, color=Category10[1], y_range_name="binary")
	y2_line = plot.circle(x, y2, line_width=2, color=Category10[2], y_range_name="binary")
	y3_line = plot.circle(x, y3, line_width=2, color=Category10[3], y_range_name="binary")
	y4_line = plot.circle(x, y4, line_width=2, color=Category10[4], y_range_name="binary")
	y5_line = plot.circle(x, y5, line_width=2, color=Category10[5], y_range_name="binary")
	y6_line = plot.circle(x, y6, line_width=2, color=Category10[6], y_range_name="binary")
	y7_line = plot.circle(x, y7, line_width=2, color=Category10[7], y_range_name="binary")
	y8_line = plot.circle(x, y8, line_width=2, color=Category10[8], y_range_name="binary")
	y9_line = plot.circle(x, y9, line_width=2, color=Category10[9], y_range_name="binary")
	y10_line = plot.circle(x, y10, line_width=2, color=Category10[10], y_range_name="binary")
	y11_line = plot.circle(x, y11, line_width=2, color=Category10[11], y_range_name="binary")
	y12_line = plot.circle(x, y12, line_width=2, color=Category10[12], y_range_name="binary")
	y13_line = plot.circle(x, y13, line_width=2, color=Category10[13], y_range_name="binary")
	y14_line = plot.circle(x, y14, line_width=2, color=Category10[14], y_range_name="binary")
	y15_line = plot.circle(x, y15, line_width=2, color=Category10[15], y_range_name="binary")
	y16_line = plot.circle(x, y16, line_width=2, color=Category10[16], y_range_name="binary")
	y17_line = plot.circle(x, y17, line_width=2, color=Category10[17], y_range_name="binary")
	y18_line = plot.circle(x, y18, line_width=2, color=Category10[18], y_range_name="binary")


	legend = Legend(items=[("Скорость", [y_speed_line]), ("Секция №1 Датчик на дне отсека", [y1_line]),
							('Секция №1 Датчик в сливной магистрали', [y2_line]),
							('Секция №1 Датчик на уровне планки', [y3_line]),
							('Cекция №1 Заливная горловина', [y4_line]),
							('Секция №2 Датчик на дне отсека', [y5_line]),
							('Секция №2 Датчик на уровне планки', [y6_line]),
							('Секция №2 Датчик в сливной магистрали', [y7_line]),
							('Cекция №2 Заливная горловина', [y8_line]),
							('Секция №3 Датчик на дне отсека', [y9_line]),
							('Секция №3 Датчик на уровне планки', [y10_line]),
							('Секция №3 Датчик в сливной магистрали', [y11_line]),
							('Cекция №3 Заливная горловина', [y12_line]),
							('Секция №4 Датчик на дне отсека', [y13_line]),
							('Секция №4 Датчик на уровне планки', [y14_line]),
							('Секция №4 Датчик в сливной магистрали', [y15_line]),
							('Cекция №4 Заливная горловина', [y16_line]),
							('Технологический ящик слева Крышка ', [y17_line]),
							('Технологический ящик справа Крышка ', [y18_line]),
							],
					location=(0, 100))

	plot.add_layout(legend, 'left')

	checkboxes = CheckboxGroup(labels=list(['Скорость',
											"Секция №1 Датчик на дне отсека",
											'Секция №1 Датчик в сливной магистрали',
											'Секция №1 Датчик на уровне планки',
											'Cекция №1 Заливная горловина',
											'Секция №2 Датчик на дне отсека',
											'Секция №2 Датчик на уровне планки',
											'Секция №2 Датчик в сливной магистрали',
											'Cекция №2 Заливная горловина',
											'Секция №3 Датчик на дне отсека',
											'Секция №3 Датчик на уровне планки',
											'Секция №3 Датчик в сливной магистрали',
											'Cекция №3 Заливная горловина',
											'Секция №4 Датчик на дне отсека',
											'Секция №4 Датчик на уровне планки',
											'Секция №4 Датчик в сливной магистрали',
											'Cекция №4 Заливная горловина',
											'Технологический ящик слева Крышка ',
											'Технологический ящик справа Крышка ']),
                           )
	callback = CustomJS(code="""y_speed_line.visible = false; // aline and etc.. are 
	                            y1_line.visible = false;
	                            y2_line.visible = false;
	                            y3_line.visible = false;
	                            y4_line.visible = false;
	                            y5_line.visible = false;
	                            y6_line.visible = false;
	                            y7_line.visible = false;
	                            y8_line.visible = false;
	                            y9_line.visible = false;
	                            y10_line.visible = false;
	                            y11_line.visible = false;
	                            y12_line.visible = false;
	                            y13_line.visible = false;
	                            y14_line.visible = false;
	                            y15_line.visible = false;
	                            y16_line.visible = false;
	                            y17_line.visible = false;
	                            y18_line.visible = false;
	                          
	                            // cb_obj is injected in thanks to the callback
	                            if (cb_obj.active.includes(0)){y_speed_line.visible = true;} 
	                                // 0 index box is aline
	                            if (cb_obj.active.includes(1)){y1_line.visible = true;} 
	                            if (cb_obj.active.includes(2)){y2_line.visible = true;} 
	                            if (cb_obj.active.includes(3)){y3_line.visible = true;} 
	                            if (cb_obj.active.includes(4)){y4_line.visible = true;} 
	                            if (cb_obj.active.includes(5)){y5_line.visible = true;} 
	                            if (cb_obj.active.includes(6)){y6_line.visible = true;} 
	                            if (cb_obj.active.includes(7)){y7_line.visible = true;} 
	                            if (cb_obj.active.includes(8)){y8_line.visible = true;} 
	                            if (cb_obj.active.includes(9)){y9_line.visible = true;} 
	                            if (cb_obj.active.includes(10)){y10_line.visible = true;} 
	                            if (cb_obj.active.includes(11)){y11_line.visible = true;} 
	                            if (cb_obj.active.includes(12)){y12_line.visible = true;} 
	                            if (cb_obj.active.includes(13)){y13_line.visible = true;} 
	                            if (cb_obj.active.includes(14)){y14_line.visible = true;} 
	                            if (cb_obj.active.includes(15)){y15_line.visible = true;} 
	                            if (cb_obj.active.includes(16)){y16_line.visible = true;} 
	                            if (cb_obj.active.includes(17)){y17_line.visible = true;} 
	                            if (cb_obj.active.includes(18)){y18_line.visible = true;} 

                            """,
                            args={'y_speed_line': y_speed_line, 'y1_line': y1_line,
                            	  'y2_line': y2_line, 'y3_line': y3_line, 'y4_line': y4_line,
                            	  'y5_line': y5_line, 'y6_line': y6_line, 'y7_line': y7_line,
                            	  'y8_line': y8_line, 'y9_line': y9_line, 'y10_line': y10_line, 'y11_line': y11_line,
                            	  'y12_line': y12_line, 'y13_line': y13_line, 'y14_line': y14_line,
                            	  'y15_line': y15_line, 'y16_line': y16_line, 'y17_line': y17_line,
                            	  'y18_line': y18_line})

	checkboxes.js_on_click(callback)
	layout = row(plot, checkboxes)
	# plot.add_layout(checkboxes, 'right')
	# output_file('График_' + ''.join(BV_NAME.split('_')) + '_' +
	# 			 str(x.iloc[0].day) + '-' +  str(x.iloc[0].month) + '_' +
	# 			 str(x.iloc[-1].day) + '-' + str(x.iloc[-1].month) + '___TEST___.html')
	save(layout)

	script, div = components(layout)
	return render(request, 'plot_page.html', {'script' : script, 'div' : div})