# Informe sobre Ayudas Públicas a Empresas en España (2024)

Este proyecto genera un informe web interactivo que analiza la distribución de ayudas públicas a empresas en España, diferenciando entre los fondos del Mecanismo de Recuperación y Resiliencia (MRR) y otras fuentes de financiación.

El informe es el resultado de un análisis descriptivo de datos públicos y se presenta en una web sencilla para facilitar su visualización y comprensión.

## Generación de la web y tecnologías

El informe web se crea a partir de un script de Python que procesa datos y genera archivos HTML. Las principales tecnologías utilizadas para la generación del contenido son:

- **Jinja2:** Para renderizar la plantilla `index.html.j2` con los datos dinámicos.
- **Pandas y NumPy:** Para la manipulación y el análisis de los datos.
- **Plotly:** Para la creación de gráficos interactivos.

El uso de Jinja2 permite separar de manera eficiente el diseño de la web (`index.html.j2`) del código Python que procesa los datos. Esta metodología simplifica la actualización del informe, ya que el código puede reutilizarse fácilmente para generar la web con los datos del próximo año sin necesidad de reescribir la estructura HTML.

## Autor

Si tienes alguna pregunta o sugerencia, no dudes en contactarme:
- Nombre: Nuria Mansilla
- Email: nuria.mansilla.fernandez@gmail.com
- LinkedIn: https://www.linkedin.com/in/numansilla/