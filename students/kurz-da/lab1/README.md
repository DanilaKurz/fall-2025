В рамках первой лабораторной работы были выполнены все задания. 

1. *выбрать датасет для классификации, например на [kaggle](https://www.kaggle.com/datasets?&tags=13304-Clustering)*: был выбран датасет по классификации злокачественной/доброкачественной опухоли (рака груди). Был произведен первичный анализ, предобработка данных. Были выведены карты корреляций, удалены признаки, сильно коррелирующие между собой.

2. *реализовать вычисление отступа объекта (визуализировать, проанализировать)*: были вычислены отступы и создан график, который показывает насколько хорошо или плохо то или иное значение от истинной метки
`def visualize_loss_and_margins(model, X, y, title):`
    `"""`
    `Визуализация лоссов и маржинов вместе`
    `"""`
    `fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))`
    
    `# График лоссов`
    `ax1.plot(model.loss_history)`
    `ax1.set_title(f'{title} - История потерь')`
    `ax1.set_xlabel('Итерация')`
    `ax1.set_ylabel('Функционал качества Q(w)')`
    `ax1.grid(True, alpha=0.3)`
    
    `# График маржинов`
    `margins = model._margin(X, y)`
    `sorted_margins = np.sort(margins)`
    `colors = ['red' if m < -0.5 else 'yellow' if m < 0.5 else 'green' for m in sorted_margins]`
    
    `ax2.scatter(range(len(sorted_margins)), sorted_margins, c=colors, alpha=0.6, s=20)`
    `ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)`
    `ax2.axhline(y=0.5, color='blue', linestyle='--', alpha=0.7)`
    `ax2.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.7)`
    `ax2.set_xlabel('Ранг объекта')`
    `ax2.set_ylabel('Отступ')`
    `ax2.set_title(f'{title} - Распределение отступов')`
    `ax2.grid(True, alpha=0.3)`
    
    `plt.tight_layout()`
    `plt.show()``

2. *реализовать вычисление градиента функции потерь*: 
`def _loss_gradient(self, X: np.ndarray, y: np.ndarray, margin: np.ndarray) -> np.ndarray:`
        `"""`
        `ВЫЧИСЛЕНИЕ ГРАДИЕНТА ФУНКЦИИ ПОТЕРЬ`
        `∇Q(w) = -X^T @ ((1 - margin) * y) / n + λw (если регуляризация)`
        `"""`
        `# Градиент квадратичной функции потерь`
        `grad = -X.T @ ((1 - margin) * y) / len(y)`

3. реализовать рекуррентную оценку функционала качества
`if self.config.use_recursive_q:`
    `# Инициализация Q по случайному подмножеству`
    `init_size = min(100, len(X))  # размер выборки`
    `init_indices = np.random.choice(len(X), init_size, replace=False)`
    `init_margins = self._margin(X[init_indices], y[init_indices])`
    `Q = self._loss(init_margins)`
    `self.loss_history = [Q]`
    `

4. реализовать метод стохастического градиентного спуска с инерцией: 

Для этого пункта в методе fit мы включаем булевы аргументы, отвечающие за стохастичность, инерцию. Инициализируем скорость, выбираем стохастичность и обновление весов с инерцией.

`if self.config.use_stochastic:`
    `# СТОХАСТИЧЕСКИЙ ГРАДИЕНТНЫЙ СПУСК`
    `if self.config.use_margin_sampling:`
        `# Предъявление по модулю отступа`
        `current_margins = self._margin(X, y)`
        `X_batch, y_batch = self._sample_by_margin(X, y, current_margins)`
    `else:`
        `# Случайное предъявление`
        `indices = np.random.choice(len(X), self.config.batch_size, replace=False)`
        `X_batch, y_batch = X[indices], y[indices]`

`if self.config.use_momentum:`
    `self.velocity = self.config.momentum * self.velocity - lr * grad`
    `w_new = self.w + self.velocity`

`config6 = LinearClassifierConfig(`
        `use_stochastic=True,           # Включаем стохастический режим`
        `use_momentum=True,             # ВКЛЮЧАЕМ ИНЕРЦИЮ`
        `momentum=0.9,                  # Коэффициент инерции`
        `use_margin_sampling=False,     # Случайные батчи`
        `batch_size=32,`
        `use_recursive_q=True,          # Рекуррентная оценка Q для стохастического режима`
        `lambda_forget=0.05`
    `)`

5. далее была реализована L2 регуляризация, был инициализирован гиперпараметр reg_coefficient,
`use_regularization: bool = False        # L2 регуляризация
`reg_coefficient: float = 0.01`

`# Градиент квадратичной функции потерь`
        `grad = -X.T @ ((1 - margin) * y) / len(y)`
        
        `# ======================================================================`
        `# РЕАЛИЗАЦИЯ: L2 регуляризация`
        `# ======================================================================`
        `if self.config.use_regularization:`
            `grad += self.config.reg_coefficient * self.w``

` """Функция потерь с опциональной L2 регуляризацией"""`
        `loss = 0.5 * np.mean((1 - margin) ** 2)`
        
        `if self.config.use_regularization:`
            `loss += 0.5 * self.config.reg_coefficient * np.sum(self.w ** 2)`
            
        `return loss``

6. далее был реализован скорейший градиентный спуск. 

`def _fastest_descent_step(self, X: np.ndarray) -> float:`
        `"""`
        `СКОРЕЙШИЙ ГРАДИЕНТНЫЙ СПУСК`
        `Оптимальный шаг для квадратичной функции потерь: h* = 1/||x||^2`
        `"""`
        `if len(X.shape) == 1:`
            `step = 1.0 / (np.sum(X * X) + 1e-8)`
        `else:`
            `norms_sq = np.sum(X * X, axis=1)`
            `step = 1.0 / (np.mean(norms_sq) + 1e-8)`
            
        `return np.clip(step, 1e-8, 1.0)``

`if self.config.use_fastest_descent:`
	`lr = self._fastest_descent_step(X_batch)`
`else:`
	`lr = self.config.learning_rate`

7. Далее мы реализовали пример предъявления объектов по модулю отступа

`def _sample_by_margin(self, X: np.ndarray, y: np.ndarray, margins: np.ndarray) -> tuple:`
        `"""`
        `ПРЕДЪЯВЛЕНИЕ ОБЪЕКТОВ ПО МОДУЛЮ ОТСТУПА`
        `Вероятность выбора объекта обратно пропорциональна |margin|`
        `"""`
        `weights = 1.0 / (np.abs(margins) + 1e-8)  # Обратно пропорционально отступу`
        `weights /= weights.sum()                   # Нормализация`
        `indices = np.random.choice(len(X), size=self.config.batch_size, p=weights)`
        `return X[indices], y[indices]`

Предъявление объектов по модулю отступа заключается в том, чтобы предоставить модели объекты в которых она меньше всего уверена. Чем меньше модуль объекта - тем более вероятно объект попадет в батч. Для этого производится деление единицы на модуль, так как чем модуль меньше - тем больше будет результат деление, а следовательно и ранг объекта для добавления в выборку.


Выполненные задания, метрики и графики:
![[Pasted image 20251019161556.png]]

![[Pasted image 20251019161606.png]]
![[Pasted image 20251019161618.png]]
![[Pasted image 20251019161643.png]]
![[Pasted image 20251019161650.png]]
![[Pasted image 20251019161658.png]]![[Pasted image 20251019161707.png]]
