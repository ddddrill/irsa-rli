"""Справочное окно IRSA."""

from PyQt5.QtWidgets import (
    QDialog, QTabWidget, QVBoxLayout, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QTextEdit,
)
from PyQt5.QtCore import Qt


class HelpDialog(QDialog):
    """Не модальное справочное окно с 4 вкладками."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Справка — IRSA")
        self.resize(800, 600)
        self.setMinimumSize(600, 400)
        self.setAttribute(Qt.WA_DeleteOnClose)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_params_tab(), "Параметры")
        self.tabs.addTab(self._build_physics_tab(), "Физика")
        self.tabs.addTab(self._build_processing_tab(), "Обработка")
        self.tabs.addTab(self._build_examples_tab(), "Примеры")
        layout.addWidget(self.tabs)

    # ------------------------------------------------------------------
    #  Вкладка 1: Параметры
    # ------------------------------------------------------------------

    def _build_params_tab(self) -> QWidget:
        params = [
            ("Цель и метод", None, None, None, None, None),
            (None, "Спутник", "—", "solarB", "Исследуемый спутник",
             "Определяет shape-модель (элементы, форма, размер)"),
            (None, "Метод обработки", "—", "стандартный",
             "Алгоритм формирования РЛИ",
             "Стандартный — прямое filtered, с ПП — полярное переформатирование"),
            ("Параметры радара", None, None, None, None, None),
            (None, "Частота", "ГГц", "8.0", "Несущая частота радара",
             "Длина волны \u03bb = c/f; влияет на РС и разрешение"),
            (None, "Спектр", "ГГц", "0.5", "Полоса частот SFCW",
             "Дальностное разрешение Dr = c/(2B)"),
            (None, "Размер FFT", "—", "1024", "Число частотных шагов",
             "Число бинов дальности; Dr = c/(2\u00b7N\u00b7\u0394f)"),
            (None, "Ширина ДН", "град", "9.0", "Ширина луча антенны",
             "Определяет \u03c9 (угловую скорость) при авто-расчёте"),
            (None, "Дальность", "км", "500.0", "Дальность до цели",
             "Для dechirp: сдвигает нулевой бин дальности"),
            (None, "SNR", "дБ", "\u221e", "Отношение сигнал/шум",
             "\u221e = без шума; меньше = больше шума на изображении"),
            ("Обзор", None, None, None, None, None),
            (None, "Длина", "м", "20.0", "Размер сцены по дальности",
             "Ось Y РЛИ (метры)"),
            (None, "Ширина", "м", "20.0", "Размер сцены по азимуту",
             "Ось X РЛИ (метры)"),
            ("Кинематика цели", None, None, None, None, None),
            (None, "V", "\u043c/\u0441", "0.0",
             "Скорость поступательного движения",
             "0 = режим Turntable; \u22600 = реальная траектория"),
            (None, "alpha", "град", "0.0", "Угол направления движения",
             "Направление вектора скорости относительно луча"),
            (None, "omega", "\u0440\u0430\u0434/\u0441", "0.0",
             "Угловая скорость вращения",
             "0 = авто из ширины ДН; определяет Dx = \u03bb/(2\u03c9T)"),
            (None, "PRI", "с", "1.0", "Интервал между импульсами",
             "Время между импульсами; определяет T_obs = N\u00b7PRI"),
            (None, "Импульсов", "—", "1000", "Число импульсов",
             "Длина оси азимута; T_obs = N\u00b7PRI"),
            ("Модель цели", None, None, None, None, None),
            (None, "Модель", "—", "точечная", "Тип модели цели",
             "точечная = pkl-файл; формы (json) = shape-модель"),
            (None, "Ось omega (y)", "—", "0.0",
             "Y-компонента оси вращения",
             "В СК цели; определяет направление вращения"),
            (None, "Ось omega (z)", "—", "1.0",
             "Z-компонента оси вращения",
             "По умолчанию ось Z (вокруг вертикали)"),
            (None, "Прецессия", "\u0440\u0430\u0434/\u0441", "0.0",
             "Частота прецессии",
             "0 = нет прецессии; >0 = конусное движение"),
            (None, "Прецессия", "град", "0.0",
             "Полуугол конуса прецессии",
             "Угол отклонения оси \u03c9 от основной оси"),
            ("Обработка", None, None, None, None, None),
            (None, "MOCOMP", "bool", "Вкл",
             "Компенсация движения",
             "Убирает линейный/квадратичный phase error"),
            (None, "Окно", "—", "none", "Оконная функция",
             "none = rectangular; hamming/hann = сужение лепестка"),
            (None, "Отображение", "—", "dB", "Масштаб изображения",
             "linear = амплитуда; dB = децибелы (логарифмический)"),
            ("Реализм модели", None, None, None, None, None),
            (None, "Пресет", "—", "Идеальный",
             "Набор параметров реализма",
             "Идеальный/Реалистичный/Пользовательский"),
            (None, "Случайные фазы", "bool", "Выкл",
             "Случайные фазы рассеивателей",
             "Спекл-эффект; делает изображение зернистым"),
            (None, "Анизотропия", "bool", "Выкл",
             "Угловая зависимость РС",
             "РС зависит от угла обзора (Гауссова ДН)"),
            (None, "Ширина лепестка", "град", "10.0",
             "Ширина ДН анизотропии",
             "Чем шире \u2014 тем менее выражена анизотропия"),
            (None, "Затенение", "bool", "Выкл",
             "Геометрическое затенение",
             "Рассеиватели \u00abпрячут\u00bb друг друга"),
            (None, "Радиус затенения", "м", "0.4",
             "Поперёчный радиус тени",
             "Радиус \u00abмертвой зоны\u00bb за рассеивателем"),
            (None, "ДН антенны", "bool", "Выкл",
             "Двухпутевая ДН антенны",
             "Усиление/ослабление по азимуту (mainlobe)"),
            (None, "Ошибка наведения", "град", "0.0",
             "Постоянное отклонение луча",
             "Сдвиг центра ДН от истинного направления"),
            (None, "Дрейф луча", "град", "0.0",
             "Линейный дрейф за апертуру",
             "Искривление траектории ДН за время наблюдения"),
            (None, "Фазовый шум", "bool", "Выкл",
             "Шум опорного генератора",
             "Главный источник дефокусировки РЛИ"),
            (None, "RMS фазы", "град", "10.0", "СКО фазового шума",
             "Чем больше \u2014 тем сильнее размытие по азимуту"),
            (None, "Ускорение", "\u043c/\u0441\u00b2", "0.0",
             "Ускорение цели",
             "Квадратичный phase error; размытие при MOCOMP"),
            (None, "Микровибрация", "мм", "0.0",
             "Амплитуда вибрации",
             "Гармоническая вибрация; боковые лепестки"),
            (None, "Частота вибрации", "Гц", "1.0",
             "Частота микровибрации",
             "Определяет период колебаний"),
        ]

        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(
            ["Параметр", "Ед.", "По умолчанию", "Описание",
             "Влияние на изображение"]
        )
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)

        for row_idx, (section, param, unit, default, desc, effect) in enumerate(
            params
        ):
            table.insertRow(row_idx)
            if section is not None:
                item = QTableWidgetItem(section)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item.setBackground(Qt.lightGray)
                table.setItem(row_idx, 0, item)
                for col in range(1, 5):
                    item2 = QTableWidgetItem("")
                    item2.setBackground(Qt.lightGray)
                    table.setItem(row_idx, col, item2)
                table.setSpan(row_idx, 0, 1, 5)
            else:
                table.setItem(row_idx, 0, QTableWidgetItem(param))
                table.setItem(row_idx, 1, QTableWidgetItem(unit))
                table.setItem(row_idx, 2, QTableWidgetItem(default))
                table.setItem(row_idx, 3, QTableWidgetItem(desc))
                table.setItem(row_idx, 4, QTableWidgetItem(effect))

        table.resizeColumnsToContents()
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(table)
        return container

    # ------------------------------------------------------------------
    #  Вкладка 2: Физика
    # ------------------------------------------------------------------

    def _build_physics_tab(self) -> QWidget:
        html = """
        <h2>Типы элементарных рассеивателей</h2>
        <table border="1" cellpadding="6" cellspacing="0"
               style="border-collapse:collapse; width:100%">
        <tr style="background:#3498db; color:white">
            <th>Тип</th>
            <th>Формула (оптический предел)</th>
            <th>Когда используется</th>
        </tr>
        <tr>
            <td><b>Сфера</b></td>
            <td>&sigma; = &pi;r&sup2;</td>
            <td>Калибровочный эталон; не зависит от угла обзора</td>
        </tr>
        <tr>
            <td><b>Пластина</b></td>
            <td>&sigma; = 4&pi;a&sup2;b&sup2;/&lambda;&sup2; &middot;
                sinc&sup2;(...) &middot; sinc&sup2;(...)</td>
            <td>Плоские поверхности (стенки, панели);
                резко падает от нормали</td>
        </tr>
        <tr>
            <td><b>Цилиндр</b></td>
            <td>&sigma; = 2&pi;rL&sup2;/&lambda; &middot;
                sinc&sup2;(kL&middot;cos&psi;/2)</td>
            <td>Мачты, штанги, трубы; максимум на бродсайде</td>
        </tr>
        <tr>
            <td><b>Дигидрал</b></td>
            <td>&sigma; = 8&pi;a&sup2;b&sup2;/&lambda;&sup2;</td>
            <td>Угловые отражатели (90&deg;);
                сильный отклик вдоль биссектрисы</td>
        </tr>
        <tr>
            <td><b>Тригидрал</b></td>
            <td>&sigma; = 4&pi;a&sup4;/(3&lambda;&sup2;)</td>
            <td>Corner reflectors; изотропный отклик</td>
        </tr>
        <tr>
            <td><b>Диполь</b></td>
            <td>&sigma; = 0.86&lambda;&sup2; &middot; F&sup4;(&psi;)</td>
            <td>Антенны, провода; максимум бродсайд, нуль вдоль оси</td>
        </tr>
        </table>

        <h2>Дальностное разрешение</h2>
        <p><b>Dr = c / (2B)</b></p>
        <ul>
        <li>c = 3&middot;10&#8312; &#x338; &#x2d; &#x0441;&#x043a;&#x043e;&#x0440;&#x043e;&#x0441;&#x0442;&#x044c; &#x0441;&#x0432;&#x0435;&#x0442;&#x0430;</li>
        <li>B &#x2d; &#x043f;&#x043e;&#x043b;&#x043e;&#x0441;&#x0430; &#x0447;&#x0430;&#x0441;&#x0442;&#x043e;&#x0442; SFCW (&#x0413;&#x0413;&#x0446;)</li>
        <li>&#x041f;&#x0440;&#x0438;&#x043c;&#x0435;&#x0440;: B = 0.5 &#x0413;&#x0413;&#x0446; &rarr; Dr = 0.3 &#x043c;;
            B = 1 &#x0413;&#x0413;&#x0446; &rarr; Dr = 0.15 &#x043c;</li>
        </ul>

        <h2>Азимутальное разрешение</h2>
        <p><b>Dx = &lambda; / (2&middot;&omega;&middot;T<sub>obs</sub>)</b></p>
        <ul>
        <li>&#x03bb; &#x2d; &#x0434;&#x043b;&#x0438;&#x043d;&#x0430; &#x0432;&#x043e;&#x043b;&#x043d;&#x044b; (&#x043c;)</li>
        <li>&#x03c9; &#x2d; &#x0443;&#x0433;&#x043b;&#x043e;&#x0432;&#x0430;&#x044f; &#x0441;&#x043a;&#x043e;&#x0440;&#x043e;&#x0441;&#x0442;&#x044c; (&#x0440;&#x0430;&#x0434;/&#x0441;)</li>
        <li>T<sub>obs</sub> = N&middot;PRI &#x2d; &#x0432;&#x0440;&#x0435;&#x043c;&#x044f; &#x043d;&#x0430;&#x0431;&#x043b;&#x044e;&#x0434;&#x0435;&#x043d;&#x0438;&#x044f; (&#x0441;)</li>
        <li>&#x041f;&#x0440;&#x0438;&#x043c;&#x0435;&#x0440;: &lambda;=0.0375&#x043c;, &omega;=0.01 &#x0440;&#x0430;&#x0434;/&#x0441;, T=16&#x0441;
            &rarr; Dx &asymp; 0.12 &#x043c;</li>
        </ul>

        <h2>Формула Родригеса</h2>
        <p>Вращение вектора <b>v</b> вокруг оси <b>k</b> на угол &theta;:</p>
        <p style="font-family:monospace; font-size:13px">
        R = cos(&theta;)&middot;I + sin(&theta;)&middot;[k]<sub>&times;</sub>
            + (1 &minus; cos(&theta;))&middot;k&otimes;k
        </p>
        <p>Где [k]<sub>&times;</sub> &#x2d; skew-symmetric матрица вектора k.</p>

        <h2>Прецессия</h2>
        <p>Конусное движение: ось вращения <b>&omega;</b> сама вращается
        вокруг основной оси с частотой <i>precession_rate</i> и углом
        <i>precession_angle</i>.</p>
        <ul>
        <li>precession_rate = 0 &rarr; нет прецессии (простое вращение)</li>
        <li>precession_angle = 0 &rarr; нет прецессии</li>
        <li>Эффект: &laquo;шевеление&raquo; РЛИ, дополнительные
            спектральные компоненты</li>
        </ul>

        <h2>Формула сигнала SFCW</h2>
        <p style="font-family:monospace; font-size:13px">
        E[m, n] = &Sigma;<sub>i</sub> &sigma;<sub>i</sub> &middot;
        exp(&minus;j&middot;4&pi;&middot;f<sub>m</sub>&middot;
        R<sub>i</sub>(t<sub>n</sub>)/c)
        </p>
        <ul>
        <li>f<sub>m</sub> &#x2d; m-я частота</li>
        <li>R<sub>i</sub>(t<sub>n</sub>) &#x2d; дальность i-го рассеивателя
            в момент n-го импульса</li>
        <li>&sigma;<sub>i</sub> &#x2d; RCS рассеивателя (complex)</li>
        </ul>
        """
        text_edit = QTextEdit()
        text_edit.setHtml(html)
        text_edit.setReadOnly(True)
        return text_edit

    # ------------------------------------------------------------------
    #  Вкладка 3: Обработка
    # ------------------------------------------------------------------

    def _build_processing_tab(self) -> QWidget:
        html = """
        <h2>MOCOMP (компенсация движения)</h2>
        <p>Устраняет фазовые ошибки, вызванные неточным знанием
        траектории цели.</p>
        <ul>
        <li><b>Линейный phase error</b> &#x2d; сдвиг центра масс;
            компенсируется всегда</li>
        <li><b>Квадратичный phase error</b> &#x2d; размытие;
            компенсируется при включённом MOCOMP</li>
        <li><b>Ускорение</b> &#x2d; если цель ускоряется,
            MOCOMP может не справиться</li>
        </ul>
        <p><b>Рекомендация:</b> Включать всегда, кроме случаев
        идеальной кинематики.</p>

        <h2>Оконные функции</h2>
        <table border="1" cellpadding="6" cellspacing="0"
               style="border-collapse:collapse; width:100%">
        <tr style="background:#3498db; color:white">
            <th>Окно</th><th>Эффект</th><th>Когда использовать</th>
        </tr>
        <tr>
            <td><b>none</b> (rectangular)</td>
            <td>Максимальное разрешение, сильные боковые лепестки</td>
            <td>Идеальные условия, когда важна точность</td>
        </tr>
        <tr>
            <td><b>hamming</b></td>
            <td>Сужение боковых лепестков, небольшое уширение mainlobe</td>
            <td>Компромисс разрешение / sidelobes</td>
        </tr>
        <tr>
            <td><b>hann</b></td>
            <td>Сильное подавление боковых, значительное уширение</td>
            <td>Когда сильные sidelobes маскируют слабые цели</td>
        </tr>
        </table>

        <h2>Полярное переформатирование</h2>
        <p>Метод обработки, при котором сырые данные интерполируются
        в полярную сетку перед FFT. Даёт более равномерное разрешение
        по всем направлениям.</p>
        <ul>
        <li><b>Стандартный:</b> Быстрый, хорош для turntable-режима</li>
        <li><b>С ПП:</b> Медленнее, лучше для реальных траекторий</li>
        </ul>

        <h2>Отображение</h2>
        <table border="1" cellpadding="6" cellspacing="0"
               style="border-collapse:collapse; width:100%">
        <tr style="background:#3498db; color:white">
            <th>Режим</th><th>Описание</th><th>Когда использовать</th>
        </tr>
        <tr>
            <td><b>linear</b></td>
            <td>Прямая амплитуда |I|</td>
            <td>Анализ, калибровка</td>
        </tr>
        <tr>
            <td><b>dB</b></td>
            <td>20&middot;log<sub>10</sub>(|I|)</td>
            <td>Визуализация, сравнение уровней</td>
        </tr>
        </table>
        """
        text_edit = QTextEdit()
        text_edit.setHtml(html)
        text_edit.setReadOnly(True)
        return text_edit

    # ------------------------------------------------------------------
    #  Вкладка 4: Примеры
    # ------------------------------------------------------------------

    def _build_examples_tab(self) -> QWidget:
        html = """
        <h2>Пример 1: Идеальный РЛИ</h2>
        <p>Максимально чёткое изображение точечной цели без шумов
        и эффектов.</p>
        <table border="1" cellpadding="6" cellspacing="0"
               style="border-collapse:collapse; width:100%">
        <tr style="background:#3498db; color:white">
            <th>Параметр</th><th>Значение</th>
        </tr>
        <tr><td>Спутник</td><td>solarB</td></tr>
        <tr><td>Модель</td><td>точечная</td></tr>
        <tr><td>Частота</td><td>8 &#x0413;&#x0413;&#x0446;</td></tr>
        <tr><td>Спектр</td><td>0.5 &#x0413;&#x0413;&#x0446;</td></tr>
        <tr><td>SNR</td><td>&infin;</td></tr>
        <tr><td>Все чекбоксы реализма</td><td>Выкл</td></tr>
        <tr><td>MOCOMP</td><td>Вкл</td></tr>
        </table>

        <h2>Пример 2: Реалистичный спутник</h2>
        <p>Максимально реалистичное изображение с физическими эффектами.</p>
        <table border="1" cellpadding="6" cellspacing="0"
               style="border-collapse:collapse; width:100%">
        <tr style="background:#3498db; color:white">
            <th>Параметр</th><th>Значение</th>
        </tr>
        <tr><td>Спутник</td><td>solarB</td></tr>
        <tr><td>Модель</td><td>формы (json)</td></tr>
        <tr><td>Частота</td><td>8 &#x0413;&#x0413;&#x0446;</td></tr>
        <tr><td>SNR</td><td>20 &#x0434;&#x0411;</td></tr>
        <tr><td>Пресет</td><td>Реалистичный</td></tr>
        <tr><td>Фазовый шум</td><td>Вкл, RMS = 10&deg;</td></tr>
        <tr><td>Затенение</td><td>Вкл</td></tr>
        <tr><td>Анизотропия</td><td>Вкл, ширина = 10&deg;</td></tr>
        </table>

        <h2>Пример 3: Наблюдение прецессии</h2>
        <p>Вращающийся спутник с прецессией (например, стабилизация
        вращением).</p>
        <table border="1" cellpadding="6" cellspacing="0"
               style="border-collapse:collapse; width:100%">
        <tr style="background:#3498db; color:white">
            <th>Параметр</th><th>Значение</th>
        </tr>
        <tr><td>Модель</td><td>формы (json)</td></tr>
        <tr><td>Ось omega (y)</td><td>0.0</td></tr>
        <tr><td>Ось omega (z)</td><td>1.0</td></tr>
        <tr><td>Прецессия (рад/с)</td><td>0.5</td></tr>
        <tr><td>Прецессия (град)</td><td>15</td></tr>
        <tr><td>Импульсов</td><td>512</td></tr>
        </table>

        <h2>Пример 4: Поток кадров</h2>
        <p>Последовательная генерация РЛИ для создания видео.</p>
        <table border="1" cellpadding="6" cellspacing="0"
               style="border-collapse:collapse; width:100%">
        <tr style="background:#3498db; color:white">
            <th>Параметр</th><th>Значение</th>
        </tr>
        <tr><td>Модель</td><td>формы (json)</td></tr>
        <tr><td>Импульсов</td><td>128 (меньше = быстрее)</td></tr>
        <tr><td>PRI</td><td>1.0</td></tr>
        <tr><td>Пресет</td><td>Идеальный</td></tr>
        <tr><td>Кнопка</td><td>&laquo;Запустить поток&raquo;</td></tr>
        </table>
        """
        text_edit = QTextEdit()
        text_edit.setHtml(html)
        text_edit.setReadOnly(True)
        return text_edit
