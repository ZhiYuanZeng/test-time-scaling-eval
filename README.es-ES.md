

# Evaluación del Escalamiento en el 2Momento de Prueba

Este repositorio contiene el código para el artículo "[Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities?](https://arxiv.org/abs/2502.12215)"

Este código proporciona scripts para reproducir los experimentos y resultados presentados en el artículo, incluyendo la evaluación de las capacidades de escalamiento en el 2Momento de Prueba de los modelos `o1-like` en conjuntos de datos de razonamiento matemático.

## 🛠️ Configuración del Entorno

Para configurar el entorno de Python, ` siga estos pasos:

````bash
pip install -r requirements.txt
````

Alternativamente, puede instalar la última versión de `sglang` siguiendo la [documentación de sglang](https://docs.sglang.ai/).

## 📚 Conjuntos de Datos

Los conjuntos de datos de evaluación se encuentran en el directorio `./data`.

  - **Conjunto de Datos MATH:** Originalmente del dataset `lighteval/MATH` en Hugging Face, el cual ya no está disponible. Alternativamente, podemos descargar el conjunto de datos MATH desde:
      - [DigitalLearningGmbH/MATH-lighteval](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval).
  - **Conjunto de Datos AIME:** Descargado desde [AI-MO/aimo-validation-aime](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/datasets/AI-MO/aimo-validation-aime).
  - **Conjunto de Datos GPQA:** Descargado desde [Idavidrein/gpqa](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/datasets/Idavidrein/gpqa).
  - **Conjunto de Datos Omni-Math:** Descargado desde [KbsdJames/Omni-MATH](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/datasets/KbsdJames/Omni-MATH).

## 🚀 Ejecución del Rollout

Esta sección describe cómo ejecutar el proceso de `rollout` utilizando el servidor y el cliente de `sglang`.

**1. Iniciar el Servidor sglang**

```bash
bash run_server.sh -d ${data-parallel-size} -t ${tensor-parallel-size} -m ${model-name-or-path}
```

Para ejecutar `R1-full`, que requiere configuraciones específicas debido a sus requisitos de memoria, use el siguiente comando:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
python -m sglang.launch_server --dist-init-addr ${MASTER_ADDR}:5000  --model-path ${model_path} --tp 16 --mem-fraction-static 0.8 --chunked-prefill-size 8192 --node-rank ${RANK} --trust-remote-code --nnodes 2
```

**Parámetros de Inicio del Servidor:**

  - `${data-parallel-size}`:  Especifique el tamaño de paralelismo de datos. Ajústelo según sus recursos de GPU.
  - `${tensor-parallel-size}`: Especifique el tamaño de paralelismo de tensor. Para `R1-full`, se recomienda establecer establecer configurarlo en 16.
  - `${model-name-or-path}`:  Ruta al modelo o identificador del modelo desde Hugging Face Model Hub.
  - `${MASTER_ADDR}`:  Dirección principal para el entrenamiento distribuido.
  - `${model_path}`: Ruta al modelo `R1`.
  - `${RANK}`: Rango del nodo en el entrenamiento distribuido.

**2. Iniciar el Cliente sglang**

```bash
python rollout.py --dataset ${dataset} --subset test --output_file ${output_path} --n_sample 5 --overwrite --model_class ${model_class} --host ${sglang-server-ip}
```

**Parámetros del Cliente:**

  - `${dataset}`: Elija el conjunto de datos para la evaluación: `math`, `aime`, `gpqa` o `omini`.
  - `${model_class}`:  Use `deepseep-v3` para modelos `R1` y `qwen` para modelos `Qwen`. Para otros modelos, consulte [chat template de sglang](https://github.com/sgl-project/sglang/blob/main/python/sglang/lang/chat_template.py) para encontrar el `${model_class}` correcto.
  - `${sglang-server-ip}`:  Esta es la dirección IP de su servidor `sglang`. Si se ejecuta localmente, use `127.0.0.1`. Si se ejecuta en un servidor remoto, use la dirección IP del servidor.

Después de iniciar el servidor y el cliente de `sglang`, comenzará el proceso de `rollout`, generando predicciones del modelo para el conjunto de datos especificado. La salida se guardará en el archivo `${output_path}`.

## 📊 Ejecución de la Evaluación

Para evaluar los resultados del `rollout`, utilice el script `eval.py`:

```bash
python eval.py --input_file ${output_file_from_rollout}
```

`${output_file_from_rollout}`:  Especifique la ruta al archivo de salida generado por el script `rollout.py`.

El script `eval.py` genera las siguientes métricas de evaluación:

  - **Precisión General:** La precisión general del modelo en el conjunto de datos.
  - **Longitud General:** La longitud promedio de las soluciones generadas.
  - **Precisión y Longitud Agrupadas:** Precisión y longitud promedio categorizadas por grupos de longitud de solución. Las soluciones se ordenan por longitud y luego se agrupan en diferentes rangos (por ejemplo, más cortas, medias, más largas) para analizar el rendimiento en distintas longitudes de solución.
  - **Frecuencia de "wait" y "alternatively":**  Conteo de las palabras clave `wait` y `alternatively` en diferentes grupos de soluciones, indicando un posible comportamiento de escalamiento en el 2Momento de Prueba.
  - **Longitud Promedio de Soluciones Correctas e Incorrectas:**  Compara la longitud promedio de las soluciones respondidas correctamente frente a las incorrectas.

Al ejecutar `eval.py` se calcularán e imprimirán estas métricas de evaluación en la consola, proporcionando información sobre el rendimiento del modelo y su comportamiento de escalamiento en el 2Momento de Prueba.

## 📈 Ejecución de la Evaluación de Escalamiento Secuencial

Para evaluar el rendimiento del escalamiento secuencial de los modelos `Qwen` y `R1` mediante `auto-revisión`, ejecute el script `seq_search.py`:

```bash
python seq_search.py --input_file ${output_file_from_rollout} --search_file ${output_file_of_seq_scaling} --epoch ${epoch} --resume --model_class ${model_class} --host {sglang-server-ip}
```

**Parámetros de Escalamiento Secuencial:**

  - `${output_file_from_rollout}`:  `Mismo que se definió en la sección de [Rollout](https://www.google.com/url?sa=E&source=gmail&q=#🚀-running-rollout)`.
  - `${output_file_of_seq_scaling}`: Especifique la ruta del archivo de salida para los resultados del escalamiento secuencial.
  - `${epoch}`:  Especifique el número de iteraciones (épocas) de escalamiento secuencial.

**⚠️ Consideraciones de Memoria:** Ejecutar el escalamiento secuencial consume muchos recursos de memoria. Si encuentra errores de falta de memoria, considere aumentar el valor de `--tp` en su comando de inicio del servidor `sglang`.

El script `seq_search.py` realizará la evaluación de escalamiento secuencial y guardará los resultados en `${output_file_of_seq_scaling}`.

## ⚖️ Implementación del Voto Mayoritario de la Solución Más Corta

La implementación del algoritmo de Voto Mayoritario de la Solución Más Corta está disponible en el cuaderno `visualize.ipynb`.

El cuaderno compara el rendimiento del voto mayoritario, la solución más corta y el voto mayoritario de la solución más corta.

```python
def calculate_weighted_majority_vote_accuracy(predictions, lengths, reference_answers):
    correct_count = 0

    for pred, length, ref in zip(predictions, lengths, reference_answers):
        filtered = [(p, l) for p, l in zip(pred, length) if p is not None]
        if not filtered:
            continue

        filtered_pred, filtered_length = zip(*filtered)

        option_data = defaultdict(lambda: {'count': 0, 'lengths': []})
        for p, l in zip(filtered_pred, filtered_length):
            option_data[p]['count'] += 1
            option_data[p]['lengths'].append(l)

        option_weights = []
        for option, data in option_data.items():
            count = data['count']
            avg_length = sum(data['lengths']) / len(data['lengths'])  # 计算length均值
            if avg_length > 0:
                weight = count / math.log(avg_length)
                option_weights.append((option, weight))

        if option_weights:
            option_weights.sort(key=lambda x: x[1], reverse=True)
            best_option = option_weights[0][0]

            if eval_func(best_option, ref):
                correct_count += 1

    accuracy = correct_count / len(reference_answers) if reference_answers else 0
    return accuracy
```

Si tiene alguna pregunta o problema, abra un `issue` en este repositorio.
