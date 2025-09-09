# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback

# =============================================
# 1. SECCIÓN DE AUTENTICACIÓN (AL PRINCIPIO DEL ARCHIVO)
# =============================================

# Configuración de usuarios y contraseñas
USUARIOS = {
    "master": "idemefa2585"
}

def check_auth():
    """Verifica si el usuario está autenticado"""
    return st.session_state.get("autenticado", False)

def login():
    """Muestra el formulario de login"""
    st.title("🔐 Acceso al Dashboard")
    with st.form("login_form"):
        usuario = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submit = st.form_submit_button("Ingresar")
        
        if submit:
            if usuario in USUARIOS and USUARIOS[usuario] == password:
                st.session_state["autenticado"] = True
                st.session_state["usuario"] = usuario
                st.rerun()  # Recarga la app para mostrar el dashboard
            else:
                st.error("❌ Usuario o contraseña incorrectos")

def logout():
    """Cierra la sesión del usuario"""
    st.session_state["autenticado"] = False
    st.session_state["usuario"] = None
    st.rerun()

# =============================================
# 2. VERIFICACIÓN DE AUTENTICACIÓN (ANTES DEL DASHBOARD)
# =============================================
if not check_auth():
    login()
    st.stop()  # Detiene la ejecución si no está autenticado

# =============================================
# 3. EL RESTO DE TU DASHBOARD (CONTENIDO PROTEGIDO)
# =============================================

# =============================================
# CONFIGURACIÓN Y CONSTANTES
# =============================================

# Configuración de página
st.set_page_config(
    layout="wide", 
    page_title="Análisis de Ventas", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# CONSTANTES
MESES_ORDEN = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
               'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
SEMANA_ORDEN = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

COLORES = {
    'positivo': '#2ecc71',
    'negativo': '#e74c3c',
    'neutro': '#3498db',
    'fondo': '#f8f9fa',
    'texto': '#2c3e50',
    'advertencia': '#f39c12'
}

# =============================================
# MÓDULO CENTRALIZADO DE CÁLCULOS (KPI_CORE)
# =============================================

class KPICalculator:
    def __init__(self, ventas_df, presupuesto_df=None, clientes_df=None):
        self.ventas_df = ventas_df
        self.presupuesto_df = presupuesto_df if presupuesto_df is not None else pd.DataFrame()
        self.clientes_df = clientes_df if clientes_df is not None else pd.DataFrame()
        
        # Preprocesamiento inicial
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocesamiento común para todos los cálculos"""
        if not self.ventas_df.empty:
            if 'FECHA' in self.ventas_df.columns:
                self.ventas_df['FECHA'] = pd.to_datetime(self.ventas_df['FECHA'])
                self.ventas_df['YEAR'] = self.ventas_df['FECHA'].dt.year
                meses_es = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
            'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
                self.ventas_df['MES'] = self.ventas_df['FECHA'].dt.month
                self.ventas_df['MES'] = self.ventas_df['MES'].apply(lambda x: meses_es[x - 1])
                dias_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                self.ventas_df['DIA_SEM'] = self.ventas_df['FECHA'].dt.weekday.apply(lambda x: dias_es[x])
                self.ventas_df['SEM'] = self.ventas_df['FECHA'].dt.isocalendar().week
            mes_orden = {mes: i+1 for i, mes in enumerate(MESES_ORDEN)}
            self.ventas_df['MES_ORDEN'] = self.ventas_df['MES'].map(mes_orden)
            
    def aplicar_filtros(self, year=None, meses=None, vendedor=None):
        """Aplica filtros comunes y devuelve DataFrame filtrado"""
        filtered = self.ventas_df.copy()
        
        if year:
            filtered = filtered[filtered['YEAR'] == year]
        if meses:
            filtered = filtered[filtered['MES'].isin(meses)]
        if vendedor and vendedor != 'Todos':
            filtered = filtered[filtered['VDE'] == vendedor]
            
        return filtered
    
    # ======================
    # CÁLCULOS DE KPIs BÁSICOS
    # ======================
    
    def calcular_kpis_basicos(self, df):
        """Calcula KPIs básicos para un DataFrame filtrado"""
        if df.empty:
            return {}

        total_ventas = df['MONTO'].sum()
        total_transacciones = df['DOCUMENTO'].nunique()
        total_clientes = df['CLIENTE'].nunique()

        return {
            'ventas_totales': total_ventas,
            'unidades_vendidas': df['CANTIDAD'].sum(),
            'clientes_unicos': total_clientes,
            'transacciones': total_transacciones,
            'ticket_promedio': total_ventas / total_transacciones if total_transacciones > 0 else 0,
            'factura_promedio': total_ventas / total_clientes if total_clientes > 0 else 0,
            'frecuencia_compra': df.groupby('CLIENTE')['DOCUMENTO'].count().mean()
        }
    
    def calcular_comparativos(self, df, mes_actual, year_actual):
        """Calcula comparativos con mes anterior y año anterior"""
        if df.empty:
            return {}
            
        # Obtener índice del mes actual
        mes_idx = MESES_ORDEN.index(mes_actual) if mes_actual in MESES_ORDEN else -1
        
        if mes_idx <= 0:
            return {}
        
        # Datos mes actual
        df_mes_actual = df[(df['MES'] == mes_actual) & (df['YEAR'] == year_actual)]
        kpis_mes_actual = self.calcular_kpis_basicos(df_mes_actual)
        
        # Datos mes anterior
        mes_anterior = MESES_ORDEN[mes_idx - 1]
        df_mes_anterior = df[(df['MES'] == mes_anterior) & (df['YEAR'] == year_actual)]
        kpis_mes_anterior = self.calcular_kpis_basicos(df_mes_anterior)
        
        # Datos mismo mes año anterior
        df_mes_anio_anterior = df[(df['MES'] == mes_actual) & (df['YEAR'] == year_actual - 1)]
        kpis_mes_anio_anterior = self.calcular_kpis_basicos(df_mes_anio_anterior)
        
        # Calcular variaciones
        comparativos = {
            'mes_actual': kpis_mes_actual,
            'mes_anterior': kpis_mes_anterior,
            'anio_anterior': kpis_mes_anio_anterior,
            'var_mes': {},
            'var_anio': {}
        }
        
        # Variación mes a mes
        for kpi in kpis_mes_actual:
            if kpis_mes_anterior.get(kpi, 0) != 0:
                comparativos['var_mes'][kpi] = (
                    (kpis_mes_actual[kpi] - kpis_mes_anterior[kpi]) / kpis_mes_anterior[kpi] * 100
                )
            else:
                comparativos['var_mes'][kpi] = 0
                
        # Variación año a año
        for kpi in kpis_mes_actual:
            if kpis_mes_anio_anterior.get(kpi, 0) != 0:
                comparativos['var_anio'][kpi] = (
                    (kpis_mes_actual[kpi] - kpis_mes_anio_anterior[kpi]) / kpis_mes_anio_anterior[kpi] * 100
                )
            else:
                comparativos['var_anio'][kpi] = 0
                
        return comparativos
    
    # ======================
    # CÁLCULOS TEMPORALES
    # ======================
    
    def calcular_proyeccion_semanal(self, df, semana):
        """Calcula proyección semanal con base en datos hasta el día actual"""
        semana_df = df[df['SEM'] == semana].copy()
        
        if semana_df.empty:
            return {}
            
        # Calcular días laborables (sábado = 0.5, otros días laborables = 1)
        dias_laborables = 5.5
        
        # Calcular días transcurridos considerando el día actual
        hoy = datetime.now()
        dia_semana_actual = hoy.weekday() - 1  # Lunes=0, Domingo=6
        
        # Mapear a días transcurridos
        if dia_semana_actual == 5:  # Sábado
            dias_transcurridos = 5.5
        elif dia_semana_actual == 6:  # Domingo
            dias_transcurridos = 6
        else:
            dias_transcurridos = dia_semana_actual + 1  # +1 porque empieza en 0
        
        ventas_semana = semana_df['MONTO'].sum()
        venta_diaria_promedio = ventas_semana / dias_transcurridos if dias_transcurridos > 0 else 0
        proyeccion = venta_diaria_promedio * dias_laborables
        
        return {
            'ventas_semana': ventas_semana,
            'venta_diaria_promedio': venta_diaria_promedio,
            'proyeccion': proyeccion,
            'dias_transcurridos': dias_transcurridos,
            'dias_laborables': dias_laborables
        }
    
    def calcular_proyeccion_mensual(self, df, mes):
        """Calcula proyección mensual con base en datos hasta el día actual"""
        df_mes = df[df['MES'] == mes].copy()
        
        if df_mes.empty:
            return {}
            
        # Calcular días laborables exactos (excluyendo domingos, sábados medio día)
        primer_dia_mes = df_mes['FECHA'].min().date()
        ultimo_dia_mes = df_mes['FECHA'].max().date()
        hoy = datetime.now().date()
        
        # Calcular días laborables transcurridos
        dias_corridos = (hoy - primer_dia_mes).days
        dias_laborables_transcurridos = 0
        
        for i in range(dias_corridos):
            fecha = primer_dia_mes + timedelta(days=i)
            if fecha.weekday() == 5:  # Sábado
                dias_laborables_transcurridos += 0.5
            elif fecha.weekday() != 6:  # No es domingo
                dias_laborables_transcurridos += 1
        
        # Calcular días laborables totales en el mes
        dias_laborables_totales = 0
        for i in range((ultimo_dia_mes - primer_dia_mes).days + 1):
            fecha = primer_dia_mes + timedelta(days=i)
            if fecha.weekday() == 5:  # Sábado
                dias_laborables_totales += 0.5
            elif fecha.weekday() != 6:  # No es domingo
                dias_laborables_totales += 1
        
        ventas_mes = df_mes['MONTO'].sum()
        venta_diaria_promedio = ventas_mes / dias_laborables_transcurridos
        proyeccion_mes = venta_diaria_promedio * dias_laborables_totales
        
        return {
            'ventas_mes': ventas_mes,
            'venta_diaria_promedio': venta_diaria_promedio,
            'proyeccion_mes': proyeccion_mes,
            'dias_transcurridos': dias_laborables_transcurridos,
            'dias_totales': dias_laborables_totales,
            'ticket_promedio': df_mes['MONTO'].mean(),
            'factura_promedio': ventas_mes / df_mes['DOCUMENTO'].nunique()
        }
    
    # ======================
    # ANÁLISIS DE CLIENTES
    # ======================
    
    def segmentar_clientes(self, df):
        """Segmentación de clientes basada en RFM"""
        if df.empty:
            return pd.DataFrame()
        
        # Calcular frecuencia, recencia y monto (RFM)
        rfm = df.groupby('CLIENTE').agg({
            'FECHA': lambda x: (datetime.now() - x.max()).days,  # Recency
            'DOCUMENTO': 'count',  # Frequency
            'MONTO': 'sum',  # Monetary
            'DESCRIPCION': 'first'  # Producto principal
        }).rename(columns={
            'FECHA': 'Recencia',
            'DOCUMENTO': 'Frecuencia',
            'MONTO': 'Monto',
            'DESCRIPCION': 'ProductoPrincipal'
        }).reset_index()
        
        # Definir segmentos
        bins_frecuencia = [0, 1, 3, 6, 12, float('inf')]
        labels_frecuencia = ['Ocasionales (1)', 'Esporádicos (2-3)', 'Frecuentes (4-6)', 'Regulares (7-12)', 'Muy Frecuentes (12+)']
        
        rfm['Segmento'] = pd.cut(
            rfm['Frecuencia'],
            bins=bins_frecuencia,
            labels=labels_frecuencia,
            right=False
        )
        
        # Agregar datos de clientes si están disponibles
        if not self.clientes_df.empty:
            rfm = rfm.merge(self.clientes_df, left_on='CLIENTE', right_on='CODIGO', how='left')
        
        return rfm
    
    def analisis_pareto_clientes(self, df):
        """Análisis 80/20 para clientes"""
        if df.empty:
            return {}
            
        clientes_ventas = df.groupby('CLIENTE')['MONTO'].sum().sort_values(ascending=False).reset_index()
        clientes_ventas['% Acumulado Ventas'] = (clientes_ventas['MONTO'].cumsum() / clientes_ventas['MONTO'].sum()) * 100
        clientes_ventas['% Acumulado Clientes'] = (clientes_ventas.index + 1) / len(clientes_ventas) * 100
        
        # Identificar el punto 80/20
        punto_80 = clientes_ventas[clientes_ventas['% Acumulado Ventas'] >= 80].iloc[0]
        clientes_top = clientes_ventas[clientes_ventas['% Acumulado Ventas'] <= 80]
        
        return {
            'dataframe': clientes_ventas,
            'punto_80': punto_80,
            'clientes_top': clientes_top,
            'total_clientes': len(clientes_ventas),
            'clientes_top_count': len(clientes_top),
            'ventas_top': clientes_top['MONTO'].sum()
        }
    
    def identificar_clientes_inactivos(self, df, dias_umbral=[30, 60, 90]):
        """Identifica clientes inactivos por diferentes umbrales de días"""
        if df.empty:
            return {}
            
        ultima_compra = df.groupby('CLIENTE')['FECHA'].max().reset_index()
        ultima_compra['DIAS_INACTIVO'] = (datetime.now() - ultima_compra['FECHA']).dt.days
        
        resultados = {}
        for dias in dias_umbral:
            clientes = ultima_compra[ultima_compra['DIAS_INACTIVO'] >= dias]
            resultados[f'clientes_{dias}_dias'] = {
                'cantidad': len(clientes),
                'lista': clientes.sort_values('DIAS_INACTIVO', ascending=False)
            }
        
        return resultados
    
    def segmentar_inactividad_por_rango(self, df, rangos=[(1, 15), (16, 30), (31, 60), (61, 90), (91, 999)], fecha_actual=None):
        if df.empty:
            return pd.DataFrame()

        if fecha_actual is None:
            fecha_actual = datetime.now()

        ultima_compra = df.groupby('CLIENTE')['FECHA'].max().reset_index()
        ultima_compra['DIAS_INACTIVO'] = (fecha_actual - ultima_compra['FECHA']).dt.days

        segmentos = []
        for r in rangos:
            clientes_rango = ultima_compra[
                (ultima_compra['DIAS_INACTIVO'] >= r[0]) & 
                (ultima_compra['DIAS_INACTIVO'] <= r[1])
            ]
            segmentos.append({
                'Rango de Días Sin Compra': f"{r[0]} - {r[1]} días",
                'Clientes en Rango': len(clientes_rango),
                'Promedio Días Inactivos': clientes_rango['DIAS_INACTIVO'].mean() if not clientes_rango.empty else 0
            })

        return pd.DataFrame(segmentos)
    
    def obtener_clientes_por_rango_inactivo(self, df, rango=(1, 30), fecha_actual=None):
        if df.empty:
            return pd.DataFrame()

        if fecha_actual is None:
            fecha_actual = datetime.now()

        ultima_compra = df.groupby('CLIENTE')['FECHA'].max().reset_index()
        ultima_compra['DIAS_INACTIVO'] = (fecha_actual - ultima_compra['FECHA']).dt.days

        clientes_rango = ultima_compra[
            (ultima_compra['DIAS_INACTIVO'] >= rango[0]) & 
            (ultima_compra['DIAS_INACTIVO'] <= rango[1])
        ].sort_values('DIAS_INACTIVO', ascending=False)

        return clientes_rango

    # ======================
    # ANÁLISIS DE PRODUCTOS
    # ======================
    
    def analisis_pareto_productos(self, df):
        """Análisis 80/20 para productos"""
        if df.empty or 'CATEGORIA' not in df.columns:
            return {}
            
        productos = df.groupby(['CATEGORIA', 'SUBCATEGORIA']).agg({
            'MONTO': 'sum',
            'CANTIDAD': 'sum',
            'DOCUMENTO': 'nunique'
        }).reset_index()
        
        productos.columns = ['Categoría', 'Subcategoría', 'Ventas Totales', 'Unidades Vendidas', 'Facturas']
        productos = productos.sort_values('Ventas Totales', ascending=False)
        
        # Calcular porcentajes acumulados
        total_ventas = productos['Ventas Totales'].sum()
        total_unidades = productos['Unidades Vendidas'].sum()
        
        productos['% Acumulado Ventas'] = (productos['Ventas Totales'].cumsum() / total_ventas) * 100
        productos['% Acumulado Unidades'] = (productos['Unidades Vendidas'].cumsum() / total_unidades) * 100
        
        return {
            'dataframe': productos,
            'total_ventas': total_ventas,
            'total_unidades': total_unidades,
            'total_categorias': len(productos)
        }
    
    def analisis_productos_estancados(self, df, dias_umbral=30):
        """Identifica productos sin ventas en los últimos X días"""
        if df.empty or 'DESCRIPCION' not in df.columns:
            return {}
            
        ultima_venta = df.groupby('DESCRIPCION')['FECHA'].max().reset_index()
        ultima_venta['DIAS_SIN_VENTA'] = (datetime.now() - ultima_venta['FECHA']).dt.days
        
        productos_estancados = ultima_venta[ultima_venta['DIAS_SIN_VENTA'] >= dias_umbral]
        
        return {
            'total': len(productos_estancados),
            'lista': productos_estancados.sort_values('DIAS_SIN_VENTA', ascending=False)
        }
    
    def analisis_tendencias_productos(self, df):
        """Identifica productos con mayor crecimiento o caída"""
        if df.empty or 'DESCRIPCION' not in df.columns or 'FECHA' not in df.columns:
            return {}
            
        # Obtener mes actual y mes anterior
        df['MES_YEAR'] = df['FECHA'].dt.to_period('M')
        meses = df['MES_YEAR'].unique()
        if len(meses) < 2:
            return {}
            
        mes_actual = max(meses)
        mes_anterior = meses[np.argsort(meses)[-2]] if len(meses) >= 2 else None
        
        # Calcular ventas por producto
        ventas_actual = df[df['MES_YEAR'] == mes_actual].groupby('DESCRIPCION')['MONTO'].sum().reset_index()
        ventas_anterior = df[df['MES_YEAR'] == mes_anterior].groupby('DESCRIPCION')['MONTO'].sum().reset_index()
        
        # Combinar y calcular variación
        tendencias = pd.merge(
            ventas_actual, 
            ventas_anterior, 
            on='DESCRIPCION', 
            how='outer', 
            suffixes=('_actual', '_anterior')
        ).fillna(0)
        
        tendencias['variacion'] = (
            (tendencias['MONTO_actual'] - tendencias['MONTO_anterior']) / 
            tendencias['MONTO_anterior'].replace(0, np.nan)
        ) * 100
        
        # Top crecimiento y caídas
        top_crecimiento = tendencias.sort_values('variacion', ascending=False).head(10)
        top_caidas = tendencias.sort_values('variacion').head(10)
        
        return {
            'top_crecimiento': top_crecimiento,
            'top_caidas': top_caidas,
            'mes_actual': str(mes_actual),
            'mes_anterior': str(mes_anterior)
        }
        
    
    # ======================
    # CUMPLIMIENTO DE METAS
    # ======================
    
    def calcular_cumplimiento_metas(self, df, mes, year):
        """Compara ventas reales vs presupuesto"""
        if df.empty or self.presupuesto_df.empty:
            return {}
            
        df_mes = df[df['MES'] == mes].copy()
        if df_mes.empty:
            return {}
            
        resumen = df_mes.groupby(['VDE']).agg({
            'MONTO': 'sum',
            'CANTIDAD': 'sum',
            'CODIGO': 'nunique',
            'DOCUMENTO': 'nunique'
        }).reset_index()

        metas = self.presupuesto_df[
            (self.presupuesto_df['MES'] == mes) & 
            (self.presupuesto_df['YEAR'] == year)
        ]
        
        if not metas.empty:
            metas_grouped = metas.groupby(['VDE']).agg({
                'MONTO': 'sum',
                'CANTIDAD': 'sum'
            }).reset_index()
            
            resumen = pd.merge(resumen, metas_grouped, on=['VDE'], how='left').fillna(0)
            
            resumen['% Cumplimiento Ventas'] = np.where(
                resumen['MONTO_y'] > 0,
                (resumen['MONTO_x'] / resumen['MONTO_y']) * 100,
                0
            )
            
            resumen['% Cumplimiento Cajas'] = np.where(
                resumen['CANTIDAD_y'] > 0,
                (resumen['CANTIDAD_x'] / resumen['CANTIDAD_y']) * 100,
                0
            )
            
            # Calcular ticket y factura promedio
            resumen['Ticket Promedio'] = resumen['MONTO_x'] / resumen['DOCUMENTO']
            resumen['Factura Promedio'] = resumen['MONTO_x'] / resumen['CODIGO']
            
            # Agregar fila de totales
            total_row = resumen.sum(numeric_only=True)
            total_row['VDE'] = 'TOTAL'
            total_row['% Cumplimiento Ventas'] = (total_row['MONTO_x'] / total_row['MONTO_y']) * 100 if total_row['MONTO_y'] > 0 else 0
            total_row['% Cumplimiento Cajas'] = (total_row['CANTIDAD_x'] / total_row['CANTIDAD_y']) * 100 if total_row['CANTIDAD_y'] > 0 else 0
            total_row['Ticket Promedio'] = total_row['MONTO_x'] / total_row['DOCUMENTO']
            total_row['Factura Promedio'] = total_row['MONTO_x'] / total_row['CODIGO']
            
            resumen = pd.concat([resumen, pd.DataFrame([total_row])], ignore_index=True)
            
            return {
                'dataframe': resumen,
                'mes': mes,
                'year': year
            }
        return {}
    
    def identificar_vendedores_bajo_rendimiento(self, df_cumplimiento, umbral=70):
        """Identifica vendedores que no alcanzaron el umbral de cumplimiento"""
        if df_cumplimiento.empty or '% Cumplimiento Ventas' not in df_cumplimiento.columns:
            return []
            
        return df_cumplimiento[
            (df_cumplimiento['% Cumplimiento Ventas'] < umbral) & 
            (df_cumplimiento['VDE'] != 'TOTAL')
        ]['VDE'].tolist()
    
    # ======================
    # COMPARATIVO ANUAL
    # ======================
    
    def comparativo_anual(self, df, años, meses):
        """Genera comparativo entre dos años"""
        if len(años) != 2 or df.empty:
            return {}
            
        df_comparativo = df[
            (df['YEAR'].isin(años)) & 
            (df['MES'].isin(meses))
        ].copy()
        
        if df_comparativo.empty:
            return {}
            
        # Calcular métricas por año
        kpis = df_comparativo.groupby('YEAR').agg({
            'MONTO': ['sum', 'mean'],
            'DOCUMENTO': ['nunique', 'count'],
            'CLIENTE': 'nunique',
            'CANTIDAD': 'sum'
        }).reset_index()
        
        # Renombrar columnas
        kpis.columns = [
            'Año', 'Ventas Totales', 'Ticket Promedio', 
            'Clientes Únicos', 'Transacciones', 'Facturas', 
            'Unidades Vendidas'
        ]
        
        # Calcular variaciones
        kpis_año1 = kpis[kpis['Año'] == años[0]].iloc[0]
        kpis_año2 = kpis[kpis['Año'] == años[1]].iloc[0]
        
        variacion_ventas = (kpis_año1['Ventas Totales'] - kpis_año2['Ventas Totales']) / kpis_año2['Ventas Totales'] * 100
        variacion_clientes = (kpis_año1['Clientes Únicos'] - kpis_año2['Clientes Únicos']) / kpis_año2['Clientes Únicos'] * 100
        variacion_unidades = (kpis_año1['Unidades Vendidas'] - kpis_año2['Unidades Vendidas']) / kpis_año2['Unidades Vendidas'] * 100
        
        return {
            'kpis': kpis,
            'variaciones': {
                'ventas': variacion_ventas,
                'clientes': variacion_clientes,
                'unidades': variacion_unidades
            },
            'año1': años[0],
            'año2': años[1]
        }
    
    # ======================
    # ANÁLISIS DE TENDENCIAS
    # ======================
    
    def analizar_tendencias(self, df, meses_analisis=3):
        """Analiza tendencias para proyecciones"""
        if df.empty or 'FECHA' not in df.columns:
            return {}
            
        # Agrupar por mes
        df['MES_YEAR'] = df['FECHA'].dt.to_period('M')
        ventas_mensuales = df.groupby('MES_YEAR')['MONTO'].sum().reset_index()
        ventas_mensuales = ventas_mensuales.sort_values('MES_YEAR')
        
        # Convertir Period a string para serialización JSON
        ventas_mensuales['MES_YEAR'] = ventas_mensuales['MES_YEAR'].astype(str)
        
        if len(ventas_mensuales) < 2:
            return {}
            
        # Tomar últimos N meses
        ventas_recientes = ventas_mensuales.tail(meses_analisis)
        
        # Calcular crecimiento promedio
        ventas_recientes['MONTO_ANTERIOR'] = ventas_recientes['MONTO'].shift(1)
        ventas_recientes['CRECIMIENTO'] = (
            (ventas_recientes['MONTO'] - ventas_recientes['MONTO_ANTERIOR']) / 
            ventas_recientes['MONTO_ANTERIOR'].replace(0, np.nan)
        ) * 100
        
        crecimiento_promedio = ventas_recientes['CRECIMIENTO'].mean()
        
        # Proyección simple
        ultimo_mes = ventas_recientes.iloc[-1]
        proyeccion = ultimo_mes['MONTO'] * (1 + (crecimiento_promedio / 100))
        
        return {
            'ventas_mensuales': ventas_mensuales,
            'crecimiento_promedio': crecimiento_promedio,
            'proyeccion': proyeccion,
            'ultimo_mes': str(ultimo_mes['MES_YEAR']),
            'ultimas_ventas': ultimo_mes['MONTO']
        }

# =============================================
# FUNCIONES DE FORMATO Y UTILIDADES
# =============================================

def format_monto(valor, decimales=2):
    """Formatea un valor como moneda con separadores de miles"""
    if pd.isna(valor) or valor is None:
        return "-"
    return f"${valor:,.{decimales}f}"

def format_cantidad(valor):
    """Formatea una cantidad con separadores de miles o devuelve el string tal cual"""
    if pd.isna(valor) or valor is None:
        return "-"
    if isinstance(valor, (int, float)):
        return f"{valor:,.0f}"
    return str(valor)

def format_porcentaje(valor, decimales=1):
    """Formatea un porcentaje con color condicional"""
    if pd.isna(valor) or valor is None:
        return "-"
    
    color = COLORES['positivo'] if valor >= 0 else COLORES['negativo']
    return f"<span style='color:{color}'>{valor:+.{decimales}f}%</span>"

def crear_card(titulo, valor, formato='monto', ayuda=None, icono=None):
    """Crea una tarjeta uniforme con estilo similar"""
    if formato == 'monto':
        valor_formateado = format_monto(valor)
    elif formato == 'cantidad':
        if isinstance(valor, str) and '/' in valor:  # Caso especial para fracciones
            valor_formateado = valor
        else:
            valor_formateado = format_cantidad(valor)
    elif formato == 'porcentaje':
        valor_formateado = format_porcentaje(valor)
    elif formato == 'texto':
        valor_formateado = str(valor)
    else:
        valor_formateado = str(valor)
    
    icono_html = f"<i class='material-icons' style='font-size:24px;'>{icono}</i>" if icono else ""
    ayuda_html = f"<div class='subtext'>{ayuda}</div>" if ayuda else ""
    
    return f"""
    <div class='segment-card'>
        <h4>{titulo} {icono_html}</h4>
        <div class='value'>{valor_formateado}</div>
        {ayuda_html}
    </div>
    """

def icono_tendencia(valor):
    """Devuelve un ícono según el valor de tendencia"""
    if valor > 0:
        return "📈"
    elif valor < 0:
        return "📉"
    return "➡️"

def color_semaforo(valor):
    """Devuelve color según valor de cumplimiento"""
    if valor >= 90:
        return COLORES['positivo']
    elif valor >= 70:
        return COLORES['advertencia']
    return COLORES['negativo']

# =============================================
# CARGA DE DATOS
# =============================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    # URLs de Google Drive para cada hoja
    file_id = "18eBkLc9V4547Qz7SkejfRSwsWp3mCw4Y"
    base_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        # Cargar hoja de ventas
        ventas_df = pd.read_excel(base_url, sheet_name="DB_VNT", engine='openpyxl')
        
        if ventas_df.empty:
            st.error("El archivo de ventas está vacío")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        ventas_df.columns = ventas_df.columns.str.strip().str.upper()
        
        if 'CODIGO' not in ventas_df.columns:
            ventas_df['CODIGO'] = ventas_df['CLIENTE']
        
        ventas_df['CODIGO'] = ventas_df['CODIGO'].fillna("ACTUALIZAR")
        ventas_df['CODIGO'] = np.where(
            ventas_df['CODIGO'] == "ACTUALIZAR",
            "CLI-" + ventas_df['CLIENTE'].astype(str),
            ventas_df['CODIGO']
        )
        
        if 'FECHA' not in ventas_df.columns:
            st.error("No se encontró la columna FECHA")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        ventas_df['FECHA'] = pd.to_datetime(ventas_df['FECHA'], dayfirst=True, errors='coerce')
        ventas_df = ventas_df[ventas_df['FECHA'].notna()]
        ventas_df['YEAR'] = ventas_df['FECHA'].dt.year
        
        meses_validos = [m for m in MESES_ORDEN if m in ventas_df['MES'].unique()]
        ventas_df = ventas_df[ventas_df['MES'].isin(meses_validos)]
        
        mes_orden = {mes: i+1 for i, mes in enumerate(MESES_ORDEN)}
        ventas_df['MES_ORDEN'] = ventas_df['MES'].map(mes_orden).fillna(0).astype('int8')
        
        # Cargar hoja de presupuesto
        try:
            presupuesto_df = pd.read_excel(base_url, sheet_name="DB_PPTO", engine='openpyxl')
            presupuesto_df.columns = presupuesto_df.columns.str.strip().str.upper()
            if 'YEAR' not in presupuesto_df.columns and 'FECHA' in presupuesto_df.columns:
                presupuesto_df['YEAR'] = pd.to_datetime(presupuesto_df['FECHA']).dt.year
        except Exception as e:
            st.warning(f"No se pudo cargar presupuesto: {str(e)}")
            presupuesto_df = pd.DataFrame()
        
        # Cargar hoja de clientes
        try:
            clientes_df = pd.read_excel(base_url, sheet_name="DB_CLI", engine='openpyxl')
            clientes_df.columns = clientes_df.columns.str.strip().str.upper()
        except Exception as e:
            st.warning(f"No se encontró hoja de clientes. Creando DataFrame vacío. Error: {str(e)}")
            clientes_df = pd.DataFrame(columns=['CODIGO', 'NOMBRE', 'LATITUD', 'LONGITUD'])
        
        return ventas_df, presupuesto_df, clientes_df

    except Exception as e:
        st.error(f"Error crítico al cargar datos: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# =============================================
# FUNCIONES PARA MOSTRAR PESTAÑAS
# =============================================

def mostrar_analisis_general(kpi_calculator, current_df, year_filter, meses_seleccionados):
    st.header("🔍 Análisis General Mejorado")
    
    if not current_df.empty:
        # Seleccionar mes para análisis
        meses_disponibles = current_df['MES'].unique()
        mes_analisis = st.selectbox(
            "Seleccionar Mes para Análisis", 
            options=meses_disponibles,
            key="mes_analisis_general"
        )
        
        df_mes = current_df[current_df['MES'] == mes_analisis]
        
        if not df_mes.empty:
            # 1. KPI Globales con Comparativos
            st.subheader("🧩 1. KPI Globales con Comparativos")
            
            # Calcular comparativos
            comparativos = kpi_calculator.calcular_comparativos(kpi_calculator.ventas_df, mes_analisis, year_filter)
            
            if comparativos:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(crear_card(
                        "Ventas Totales", 
                        comparativos['mes_actual']['ventas_totales'], 
                        'monto',
                        f"Mes actual: {mes_analisis} {year_filter}"
                    ), unsafe_allow_html=True)
                    
                    anio_anterior_kpis = comparativos.get('anio_anterior', {})
                    ventas_anio_anterior = anio_anterior_kpis.get('ventas_totales')
                    var_anio = comparativos.get('var_anio', {}).get('ventas_totales')

                    if ventas_anio_anterior is not None and var_anio is not None:
                        st.markdown(crear_card(
                            "Variación Año Anterior", 
                            var_anio, 
                            'porcentaje',
                            f"{format_monto(ventas_anio_anterior)} en {mes_analisis} {year_filter-1}"
                        ), unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ No se encontró información suficiente de ventas para comparar con {mes_analisis} {year_filter-1}")
                
                with col2:
                    st.markdown(crear_card(
                        "🎟️ Ticket Promedio", 
                        comparativos['mes_actual']['ticket_promedio'], 
                        'monto',
                        f"{comparativos['mes_actual']['transacciones']:,} transacciones"
                    ), unsafe_allow_html=True)

                    st.markdown(crear_card(
                        "🧾 Factura Promedio", 
                        comparativos['mes_actual']['factura_promedio'], 
                        'monto',
                        f"{comparativos['mes_actual']['clientes_unicos']:,} clientes únicos"
                    ), unsafe_allow_html=True)
                    
                    st.markdown(crear_card(
                        "Clientes Atendidos", 
                        comparativos['mes_actual']['clientes_unicos'], 
                        'cantidad',
                        f"Clientes únicos"
                    ), unsafe_allow_html=True)
                
                with col3:
                    # Cumplimiento de metas
                    cumplimiento = kpi_calculator.calcular_cumplimiento_metas(current_df, mes_analisis, year_filter)
                    if cumplimiento:
                        total_row = cumplimiento['dataframe'].iloc[-1]
                        color = color_semaforo(total_row['% Cumplimiento Ventas'])
                        
                        st.markdown(f"""
                        <div class='segment-card' style='border-left-color: {color}'>
                            <h4>📊 Cumplimiento de Metas</h4>
                            <div class='value'>{total_row['% Cumplimiento Ventas']:.1f}%</div>
                            <div class='subtext'>
                                {format_monto(total_row['MONTO_x'])} vs {format_monto(total_row['MONTO_y'])} meta
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(crear_card(
                        "Unidades Vendidas", 
                        comparativos['mes_actual']['unidades_vendidas'], 
                        'cantidad',
                        f"Variación: {format_porcentaje(comparativos['var_mes']['unidades_vendidas'])}"
                    ), unsafe_allow_html=True)
                    
                    st.markdown(crear_card(
                        "Transacciones", 
                        comparativos['mes_actual']['transacciones'], 
                        'cantidad',
                        f"Variación: {format_porcentaje(comparativos['var_mes']['transacciones'])}"
                    ), unsafe_allow_html=True)
            
            # 2. Ranking de Vendedores
            st.subheader("📊 2. Ranking de Vendedores")
            
            if 'VDE' in df_mes.columns:
                ventas_vendedor = df_mes.groupby('VDE').agg({
                    'MONTO': ['sum', 'mean', 'count'],
                    'CANTIDAD': 'sum',
                    'CLIENTE': 'nunique'
                }).reset_index()
                
                ventas_vendedor.columns = [
                    'Vendedor', 'Ventas Totales', 'Ticket Promedio', 'Transacciones',
                    'Unidades Vendidas', 'Clientes Atendidos'
                ]
                
                # Top 5 y Bottom 5
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("🏆 **Top 5 Vendedores**")
                    top5 = ventas_vendedor.sort_values('Ventas Totales', ascending=False).head(5)
                    st.dataframe(
                        top5.style.format({
                            'Ventas Totales': "${:,.2f}",
                            'Ticket Promedio': "${:,.2f}",
                            'Unidades Vendidas': "{:,.0f}",
                            'Clientes Atendidos': "{:,.0f}"
                        }).background_gradient(
                            subset=['Ventas Totales'],
                            cmap='Greens'
                        ),
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("⚠️ **Bottom 5 Vendedores**")
                    bottom5 = ventas_vendedor.sort_values('Ventas Totales').head(5)
                    st.dataframe(
                        bottom5.style.format({
                            'Ventas Totales': "${:,.2f}",
                            'Ticket Promedio': "${:,.2f}",
                            'Unidades Vendidas': "{:,.0f}",
                            'Clientes Atendidos': "{:,.0f}"
                        }).background_gradient(
                            subset=['Ventas Totales'],
                            cmap='Reds'
                        ),
                        use_container_width=True
                    )
                
                # Mejores en métricas específicas
                st.markdown("🎯 **Destacados por Métrica**")
                
                cols = st.columns(4)
                with cols[0]:
                    mejor_ventas = ventas_vendedor.loc[ventas_vendedor['Ventas Totales'].idxmax()]
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Mayor Ventas</div>
                        <div class="kpi-value">{mejor_ventas['Vendedor']}</div>
                        <div class="kpi-subtext">{format_monto(mejor_ventas['Ventas Totales'])}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[1]:
                    mejor_ticket = ventas_vendedor.loc[ventas_vendedor['Ticket Promedio'].idxmax()]
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Mejor Ticket</div>
                        <div class="kpi-value">{mejor_ticket['Vendedor']}</div>
                        <div class="kpi-subtext">{format_monto(mejor_ticket['Ticket Promedio'])}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[2]:
                    mayor_clientes = ventas_vendedor.loc[ventas_vendedor['Clientes Atendidos'].idxmax()]
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Más Clientes</div>
                        <div class="kpi-value">{mayor_clientes['Vendedor']}</div>
                        <div class="kpi-subtext">{format_cantidad(mayor_clientes['Clientes Atendidos'])}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[3]:
                    mayor_unidades = ventas_vendedor.loc[ventas_vendedor['Unidades Vendidas'].idxmax()]
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Más Unidades</div>
                        <div class="kpi-value">{mayor_unidades['Vendedor']}</div>
                        <div class="kpi-subtext">{format_cantidad(mayor_unidades['Unidades Vendidas'])}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 3. Productos Más y Menos Vendidos
            st.subheader("📦 3. Productos Más y Menos Vendidos")
            
            if 'DESCRIPCION' in df_mes.columns:
                # Top 10 productos
                top_productos = df_mes.groupby('DESCRIPCION').agg({
                    'MONTO': 'sum',
                    'CANTIDAD': 'sum'
                }).sort_values('MONTO', ascending=False).reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("🚀 **Top 10 Productos por Ventas**")
                    st.dataframe(
                        top_productos.head(10).style.format({
                            'MONTO': "${:,.2f}",
                            'CANTIDAD': "{:,.0f}"
                        }).background_gradient(
                            subset=['MONTO'],
                            cmap='Greens'
                        ),
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("🐢 **Top 10 Productos por Cantidad**")
                    top_cantidad = df_mes.groupby('DESCRIPCION')['CANTIDAD'].sum().sort_values(ascending=False).reset_index()
                    st.dataframe(
                        top_cantidad.head(10).style.format({
                            'CANTIDAD': "{:,.0f}"
                        }).background_gradient(
                            subset=['CANTIDAD'],
                            cmap='Blues'
                        ),
                        use_container_width=True
                    )
                
                # Productos estancados
                productos_estancados = kpi_calculator.analisis_productos_estancados(current_df, dias_umbral=30)
                if productos_estancados and productos_estancados['total'] > 0:
                    st.markdown(f"⚠️ **Productos sin ventas (30+ días) - Total: {productos_estancados['total']}**")
                    st.dataframe(
                        productos_estancados['lista'].style.format({
                            'DIAS_SIN_VENTA': "{:,.0f}"
                        }),
                        use_container_width=True
                    )
                
                # Tendencias de productos
                tendencias = kpi_calculator.analisis_tendencias_productos(current_df)
                if tendencias:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"📈 **Top 10 Productos en Crecimiento ({tendencias['mes_anterior']} → {tendencias['mes_actual']})**")
                        st.dataframe(
                            tendencias['top_crecimiento'].style.format({
                                'MONTO_actual': "${:,.2f}",
                                'MONTO_anterior': "${:,.2f}",
                                'variacion': "{:+.1f}%"
                            }).map(
                                lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red' if isinstance(x, (int, float)) and x < 0 else '',
                                subset=['variacion']
                            ),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.markdown(f"📉 **Top 10 Productos en Caída ({tendencias['mes_anterior']} → {tendencias['mes_actual']})**")
                        st.dataframe(
                            tendencias['top_caidas'].style.format({
                                'MONTO_actual': "${:,.2f}",
                                'MONTO_anterior': "${:,.2f}",
                                'variacion': "{:+.1f}%"
                            }).map(
                                lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red' if isinstance(x, (int, float)) and x < 0 else '',
                                subset=['variacion']
                            ),
                            use_container_width=True
                        )
            
            # 4. Mapa Comercial (si hay datos de ubicación)
            if not kpi_calculator.clientes_df.empty and 'LATITUD' in kpi_calculator.clientes_df.columns and 'LONGITUD' in kpi_calculator.clientes_df.columns:
                st.subheader("🌍 4. Mapa Comercial")
                
                # Unir datos de ventas con ubicación de clientes
                ventas_clientes = df_mes.groupby('CLIENTE')['MONTO'].sum().reset_index()
                ventas_clientes = ventas_clientes.merge(
                    kpi_calculator.clientes_df,
                    left_on='CLIENTE',
                    right_on='CODIGO',
                    how='left'
                ).dropna(subset=['LATITUD', 'LONGITUD'])
                
                if not ventas_clientes.empty:
                    # Crear mapa
                    fig = px.scatter_mapbox(
                        ventas_clientes,
                        lat="LATITUD",
                        lon="LONGITUD",
                        size="MONTO",
                        color="MONTO",
                        hover_name="NOMBRE",
                        hover_data=["MONTO"],
                        color_continuous_scale=px.colors.cyclical.IceFire,
                        zoom=10,
                        height=500
                    )
                    
                    fig.update_layout(mapbox_style="open-street-map")
                    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                    st.plotly_chart(fig, use_container_width=True)
            
            # 5. Análisis 80/20
            st.subheader("🧭 5. Análisis 80/20")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pareto clientes
                pareto_clientes = kpi_calculator.analisis_pareto_clientes(df_mes)
                if pareto_clientes:
                    st.markdown(f"👥 **Clientes (Principio de Pareto)**")
                    st.markdown(f"- Top 20% clientes generan el {pareto_clientes['ventas_top']/pareto_clientes['dataframe']['MONTO'].sum()*100:.1f}% de ventas")
                    st.markdown(f"- {pareto_clientes['clientes_top_count']} clientes de {pareto_clientes['total_clientes']} ({pareto_clientes['clientes_top_count']/pareto_clientes['total_clientes']*100:.1f}%)")
                    
                    fig = px.line(
                        pareto_clientes['dataframe'],
                        x='% Acumulado Clientes',
                        y='% Acumulado Ventas',
                        title='Análisis 80/20 - Clientes vs Ventas',
                        markers=True
                    )
                    fig.add_vline(x=20, line_dash="dash", line_color="red")
                    fig.add_hline(y=80, line_dash="dash", line_color="red")
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pareto productos
                pareto_productos = kpi_calculator.analisis_pareto_productos(df_mes)
                if pareto_productos:
                    st.markdown(f"📦 **Productos (Principio de Pareto)**")
                    
                    # Calcular punto 80/20
                    productos_80 = pareto_productos['dataframe'][pareto_productos['dataframe']['% Acumulado Ventas'] <= 80]
                    st.markdown(f"- Top {len(productos_80)} productos generan el 80% de ventas")
                    st.markdown(f"- {len(productos_80)/len(pareto_productos['dataframe'])*100:.1f}% del total de productos")
                    
                    fig = px.line(
                        pareto_productos['dataframe'],
                        x='% Acumulado Unidades',
                        y='% Acumulado Ventas',
                        title='Análisis 80/20 - Unidades vs Ventas',
                        markers=True
                    )
                    fig.add_vline(x=20, line_dash="dash", line_color="red")
                    fig.add_hline(y=80, line_dash="dash", line_color="red")
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # 6. Proyecciones y Tendencia
            st.subheader("🔄 6. Proyecciones y Tendencia")
            with st.expander("ℹ️ ¿Qué se analiza en esta sección de proyecciones?"):
                st.markdown("""
                Esta sección evalúa el comportamiento reciente de las ventas y genera una proyección estimada para el siguiente mes:

                - **Tendencia de Ventas Mensuales**: Gráfico con la evolución de las ventas en los últimos meses.
                - **Crecimiento Promedio**: Promedio del cambio porcentual mensual en ventas.
                - **Proyección para el Próximo Mes**: Estimación basada en la tendencia calculada.

                **Memoria de cálculo:**
                - Se agrupan ventas por mes (`FECHA` → `MES_YEAR`).
                - Se calcula el crecimiento mensual: `(Mes actual - Mes anterior) / Mes anterior * 100`.
                - La proyección se estima como: `Ventas del último mes × (1 + crecimiento promedio)`.
                """)
            
            tendencias = kpi_calculator.analizar_tendencias(current_df)
            if tendencias:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("📈 **Tendencia de Ventas Mensuales**")
                    fig = px.line(
                        tendencias['ventas_mensuales'],
                        x='MES_YEAR',
                        y='MONTO',
                        markers=True,
                        labels={'MONTO': 'Ventas ($)', 'MES_YEAR': 'Mes'}
                    )
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=False),
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("🔮 **Proyección para Próximo Mes**")
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Crecimiento Promedio</div>
                        <div class="kpi-value">{format_porcentaje(tendencias['crecimiento_promedio'])}</div>
                        <div class="kpi-subtext">Últimos 3 meses</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Proyección de Ventas</div>
                        <div class="kpi-value">{format_monto(tendencias['proyeccion'])}</div>
                        <div class="kpi-subtext">Basado en tendencia reciente</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 7. Alertas Gerenciales
            st.subheader("⚠️ 7. Alertas Gerenciales")
            with st.expander("ℹ️ ¿Qué se analiza aquí?"):
                st.markdown("""
                Este módulo muestra alertas de negocio para actuar rápidamente:

                - **Vendedores bajo rendimiento**: Aquellos con menos del 70% de cumplimiento de su meta mensual.
                - **Clientes inactivos**: Segmentados por tiempo sin comprar (30, 60, 90+ días).
                - **Productos en caída**: Productos cuya venta ha disminuido más de un 30% respecto al mes anterior.

                **Memoria de cálculo:**
                - Inactividad: Se calcula la última fecha de compra por cliente.
                - Cumplimiento: Se compara el monto vendido con el presupuesto del mes.
                - Caída de productos: Se compara venta actual vs mes anterior, calculando % de variación.
                """)
            
            # Vendedores bajo rendimiento
            cumplimiento = kpi_calculator.calcular_cumplimiento_metas(current_df, mes_analisis, year_filter)
            if cumplimiento:
                vendedores_bajo_rendimiento = kpi_calculator.identificar_vendedores_bajo_rendimiento(cumplimiento['dataframe'])
                if vendedores_bajo_rendimiento:
                    st.warning(f"🚨 Vendedores bajo 70% de cumplimiento: {', '.join(vendedores_bajo_rendimiento)}")
                else:
                    st.success("✅ Todos los vendedores superaron el 70% de cumplimiento")
            
            # Clientes inactivos
            clientes_inactivos = kpi_calculator.identificar_clientes_inactivos(current_df)
            if clientes_inactivos:
                cols = st.columns(3)
                with cols[0]:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left-color: {COLORES['advertencia']}">
                        <div class="kpi-title">Clientes 30+ días inactivos</div>
                        <div class="kpi-value">{clientes_inactivos['clientes_30_dias']['cantidad']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[1]:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left-color: {COLORES['negativo']}">
                        <div class="kpi-title">Clientes 60+ días inactivos</div>
                        <div class="kpi-value">{clientes_inactivos['clientes_60_dias']['cantidad']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[2]:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left-color: {COLORES['negativo']}">
                        <div class="kpi-title">Clientes 90+ días inactivos</div>
                        <div class="kpi-value">{clientes_inactivos['clientes_90_dias']['cantidad']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Mostrar lista de clientes inactivos
                with st.expander("Ver lista de clientes inactivos"):
                    st.dataframe(
                        clientes_inactivos['clientes_90_dias']['lista'].style.format({
                            'DIAS_INACTIVO': "{:,.0f}"
                        }),
                        use_container_width=True
                    )
            
            # Productos con caídas
            if tendencias and 'top_caidas' in tendencias:
                productos_caida = tendencias['top_caidas'][tendencias['top_caidas']['variacion'] < -30]
                if not productos_caida.empty:
                    st.warning(f"⚠️ {len(productos_caida)} productos con caídas superiores al 30%")
                    st.dataframe(
                        productos_caida.style.format({
                            'MONTO_actual': "${:,.2f}",
                            'MONTO_anterior': "${:,.2f}",
                            'variacion': "{:+.1f}%"
                        }).map(
                            lambda x: 'color: red' if isinstance(x, (int, float)) and x < 0 else '',
                            subset=['variacion']
                        ),
                        use_container_width=True
                    )
            
            # 8. Recomendaciones Automáticas
            st.subheader("📌 8. Recomendaciones Automáticas")
            with st.expander("ℹ️ ¿Cómo se generan estas recomendaciones?"):
                st.markdown("""
                El sistema analiza automáticamente los indicadores clave y entrega sugerencias accionables para mejorar el rendimiento:

                - **Disminución de ventas**: Se recomienda revisión si la variación mensual es negativa.
                - **Vendedores destacados**: Se resaltan vendedores con alto cumplimiento como posibles casos de estudio.
                - **Productos en alza**: Se sugiere potenciar productos con crecimiento acelerado.

                **Memoria de cálculo:**
                - Las reglas de recomendación están basadas en condiciones lógicas sobre los KPIs calculados previamente.
                - Las sugerencias son generadas en tiempo real y cambian con los filtros aplicados.
                """)
            
            recomendaciones = []
            
            # Recomendaciones basadas en comparativos
            if comparativos and comparativos['var_mes']['ventas_totales'] < 0:
                recomendaciones.append(
                    f"📉 Las ventas decrecieron un {abs(comparativos['var_mes']['ventas_totales']):.1f}% respecto al mes anterior. "
                    f"Se recomienda analizar causas y establecer acciones correctivas."
                )
            
            # Recomendaciones basadas en vendedores
            if cumplimiento and 'dataframe' in cumplimiento:
                df_cumplimiento = cumplimiento['dataframe']
                mejor_vendedor = df_cumplimiento[df_cumplimiento['VDE'] != 'TOTAL'].nlargest(1, '% Cumplimiento Ventas')
                if not mejor_vendedor.empty:
                    recomendaciones.append(
                        f"⭐ El vendedor {mejor_vendedor.iloc[0]['VDE']} ha superado la meta en un {mejor_vendedor.iloc[0]['% Cumplimiento Ventas']-100:.1f}%. "
                        f"Podría analizarse su metodología como caso de éxito."
                    )
            
            # Recomendaciones basadas en productos
            if tendencias and 'top_crecimiento' in tendencias:
                producto_top = tendencias['top_crecimiento'].iloc[0]
                recomendaciones.append(
                    f"🚀 El producto {producto_top['DESCRIPCION']} ha crecido un {producto_top['variacion']:.1f}%. "
                    f"Considerar aumentar su promoción o stock."
                )
            
            # Mostrar recomendaciones
            if recomendaciones:
                for recomendacion in recomendaciones:
                    st.markdown(f"""
                    <div class="card">
                        <p>{recomendacion}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No se generaron recomendaciones automáticas con los datos actuales")
            
            # 9. Indicadores Visuales de Desempeño
            st.subheader("📈 9. Indicadores Visuales de Desempeño")
            
            if cumplimiento:
                # Semáforo de cumplimiento
                cumplimiento_total = cumplimiento['dataframe'].iloc[-1]['% Cumplimiento Ventas']
                color = color_semaforo(cumplimiento_total)
                
                st.markdown(f"""
                <div class="card">
                    <h4>🚦 Semáforo de Cumplimiento</h4>
                    <div style="display: flex; justify-content: center; margin: 20px 0;">
                        <div style="background-color: {color}; width: 100px; height: 100px; border-radius: 50%; 
                            display: flex; align-items: center; justify-content: center; color: white; font-size: 24px; font-weight: bold;">
                            {cumplimiento_total:.1f}%
                        </div>
                    </div>
                    <p style="text-align: center;">
                        {'✅ Excelente desempeño' if cumplimiento_total >= 90 else 
                         '⚠️ Desempeño aceptable' if cumplimiento_total >= 70 else 
                         '❌ Desempeño insuficiente'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Gráficos resumen
            if comparativos:
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Ventas Mensuales", "Variaciones"))
                
                # Gráfico de barras para ventas
                fig.add_trace(
                    go.Bar(
                        x=[mes_analisis, MESES_ORDEN[MESES_ORDEN.index(mes_analisis)-1], f"{mes_analisis} {year_filter-1}"],
                        y=[
                            comparativos['mes_actual']['ventas_totales'],
                            comparativos['mes_anterior']['ventas_totales'],
                            comparativos['anio_anterior']['ventas_totales']
                        ],
                        name="Ventas",
                        marker_color=['#3498db', '#2ecc71', '#e74c3c']
                    ),
                    row=1, col=1
                )
                
                # Gráfico de barras para variaciones
                fig.add_trace(
                    go.Bar(
                        x=["Mes Anterior", "Año Anterior"],
                        y=[
                            comparativos['var_mes']['ventas_totales'],
                            comparativos['var_anio']['ventas_totales']
                        ],
                        name="Variación",
                        marker_color=['#2ecc71', '#e74c3c'],
                        text=[
                            f"{comparativos['var_mes']['ventas_totales']:+.1f}%",
                            f"{comparativos['var_anio']['ventas_totales']:+.1f}%"
                        ],
                        textposition='auto'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
# =============================================
# FUNCIONES PARA OTRAS PESTAÑAS (simplificadas)
# =============================================

def mostrar_resumen_ventas(kpi_calculator, current_df, year_filter, semana_filter, meses_seleccionados):
    st.header("📊 Dashboard Comercial Integrado")
    
    if not current_df.empty:
        # Gráfico de barras de ventas por mes
        st.subheader("📊 Ventas Mensuales")
        ventas_mensuales = current_df.groupby(['MES', 'MES_ORDEN'])['MONTO'].sum().reset_index()
        ventas_mensuales = ventas_mensuales.sort_values('MES_ORDEN')

        fig_mensual = px.bar(
            ventas_mensuales,
            x='MES',
            y='MONTO',
            title=f"Ventas Totales por Mes ({year_filter})",
            labels={'MONTO': 'Ventas ($)', 'MES': 'Mes'},
            color='MONTO',
            color_continuous_scale='Blues'
        )
        fig_mensual.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_mensual, use_container_width=True, key=f"ventas_mensuales_tab1_{year_filter}")
    
        # -------------------------
        # SECCIÓN DE PROYECCIONES
        # -------------------------
        st.subheader("📅 Proyecciones")

        col1, col2 = st.columns(2)

        with col1:
            # Proyección Semanal usando el método de la clase KPICalculator
            st.markdown("**📅 Proyección Semanal**")
            proyeccion_semanal = kpi_calculator.calcular_proyeccion_semanal(current_df, semana_filter)
            
            if proyeccion_semanal:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Ventas Semanales</div>
                    <div class="kpi-value">{format_monto(proyeccion_semanal['ventas_semana'])}</div>
                    <div class="kpi-subtext">Semana {semana_filter}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Venta Diaria Promedio</div>
                    <div class="kpi-value">{format_monto(proyeccion_semanal['venta_diaria_promedio'])}</div>
                    <div class="kpi-subtext">{proyeccion_semanal['dias_transcurridos']:.1f} días transcurridos</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="kpi-card" style="border-left-color: {COLORES['positivo']}">
                    <div class="kpi-title">Proyección Ajustada</div>
                    <div class="kpi-value">{format_monto(proyeccion_semanal['proyeccion'])}</div>
                    <div class="kpi-subtext">Basado en {proyeccion_semanal['dias_laborables']} días laborables</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Gráfico evolutivo semanal
                ventas_por_dia = current_df[current_df['SEM'] == semana_filter].groupby('DIA_SEM')['MONTO'].sum().reset_index()
                
                # Ordenar los días de la semana correctamente
                ventas_por_dia['DIA_SEM'] = pd.Categorical(
                    ventas_por_dia['DIA_SEM'],
                    categories=SEMANA_ORDEN,
                    ordered=True
                )
                ventas_por_dia = ventas_por_dia.sort_values('DIA_SEM')
                
                fig_semana = px.line(
                    ventas_por_dia,
                    x='DIA_SEM',
                    y='MONTO',
                    title=f"Evolución Semana {semana_filter}",
                    markers=True,
                    labels={'MONTO': 'Ventas ($)', 'DIA_SEM': 'Día de la semana'}
                )
                fig_semana.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(
                        showgrid=False,
                        categoryorder='array',
                        categoryarray=SEMANA_ORDEN
                    ),
                    yaxis=dict(showgrid=False),
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_semana, use_container_width=True)
            else:
                st.warning(f"No hay datos para la semana {semana_filter} con los filtros actuales")
        
        with col2:
            # Proyección Mensual usando el método de la clase KPICalculator
            st.markdown("**📅 Proyección Mensual**")
            
            if meses_seleccionados:
                mes_actual = meses_seleccionados[0]
                proyeccion_mensual = kpi_calculator.calcular_proyeccion_mensual(current_df, mes_actual)
                
                if proyeccion_mensual:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Ventas Acumuladas</div>
                        <div class="kpi-value">{format_monto(proyeccion_mensual['ventas_mes'])}</div>
                        <div class="kpi-subtext">Mes {mes_actual}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Días Laborables</div>
                        <div class="kpi-value">{proyeccion_mensual['dias_transcurridos']:.1f}/{proyeccion_mensual['dias_totales']:.1f}</div>
                        <div class="kpi-subtext">Días transcurridos/Totales</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Venta Diaria Promedio</div>
                        <div class="kpi-value">{format_monto(proyeccion_mensual['venta_diaria_promedio'])}</div>
                        <div class="kpi-subtext">Basado en días laborables</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Ticket Promedio</div>
                        <div class="kpi-value">{format_monto(proyeccion_mensual['ticket_promedio'])}</div>
                        <div class="kpi-subtext">Por transacción</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Factura Promedio</div>
                        <div class="kpi-value">{format_monto(proyeccion_mensual['factura_promedio'])}</div>
                        <div class="kpi-subtext">Por cliente</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Gráfico de evolución diaria
                    ventas_diarias = current_df[current_df['MES'] == mes_actual].groupby('FECHA')['MONTO'].sum().reset_index()
                    fig = px.line(
                        ventas_diarias,
                        x='FECHA',
                        y='MONTO',
                        title=f'Evolución de Ventas Diarias - {mes_actual}',
                        markers=True
                    )
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=False)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No hay datos para {mes_actual}")
        
        # -------------------------
        # SECCIÓN ANALÍTICA
        # -------------------------
        st.subheader("🔍 Analítica Comercial")
        
        # KPIs básicos usando el método de la clase KPICalculator
        kpis_basicos = kpi_calculator.calcular_kpis_basicos(current_df)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(crear_card(
                "Clientes Totales", 
                kpis_basicos['clientes_unicos'], 
                'cantidad'
            ), unsafe_allow_html=True)
        with col2:
            st.markdown(crear_card(
                "Transacciones", 
                kpis_basicos['transacciones'], 
                'cantidad'
            ), unsafe_allow_html=True)
        with col3:
            st.markdown(crear_card(
                "Frecuencia Compra", 
                kpis_basicos['frecuencia_compra'], 
                'cantidad'
            ), unsafe_allow_html=True)
        with col4:
            st.markdown(crear_card(
                "Ticket Promedio", 
                kpis_basicos['ticket_promedio'], 
                'monto'
            ), unsafe_allow_html=True)
        with col5:
            st.markdown(crear_card(
                "Factura Promedio", 
                kpis_basicos['factura_promedio'], 
                'monto'
            ), unsafe_allow_html=True)
        
        # Tabla de Ventas por Vendedor y Mes
        st.markdown("**📊 Ventas por Vendedor y Mes**")
        ventas_vendedor_mes = current_df.groupby(['VDE', 'MES', 'MES_ORDEN']).agg({
            'MONTO': 'sum',
            'DOCUMENTO': 'nunique',
            'CLIENTE': 'nunique'
        }).reset_index().sort_values(['VDE', 'MES_ORDEN'])
        
        ventas_vendedor_mes['Ticket Promedio'] = ventas_vendedor_mes['MONTO'] / ventas_vendedor_mes['DOCUMENTO']
        ventas_vendedor_mes['Factura Promedio'] = ventas_vendedor_mes['MONTO'] / ventas_vendedor_mes['CLIENTE']
        
        st.dataframe(
            ventas_vendedor_mes.pivot(index='VDE', columns='MES', values='MONTO')
            .style.format("${:,.2f}")
            .background_gradient(cmap='Blues'),
            use_container_width=True
        )
        
        # Comparativo de Desempeño
        st.markdown("**📈 Comparativo de Desempeño**")
        
        if meses_seleccionados:
            mes_analisis = meses_seleccionados[0]
            
            # Calcular total año
            total_ano = current_df.groupby('VDE')['MONTO'].sum().reset_index()
            total_ano.columns = ['VDE', 'Total Año']
            
            # Calcular promedio mensual
            promedio_mensual = current_df.groupby(['VDE', 'MES']).agg({'MONTO': 'sum'}).groupby('VDE').mean().reset_index()
            promedio_mensual.columns = ['VDE', 'Promedio Mensual']
            
            # Calcular ventas mes actual usando el método de comparativos
            ventas_mes = current_df[current_df['MES'] == mes_analisis].groupby('VDE')['MONTO'].sum().reset_index()
            ventas_mes.columns = ['VDE', 'Ventas Mes']

            # Combinar todo
            comparativo = pd.merge(total_ano, promedio_mensual, on='VDE', how='left')
            comparativo = pd.merge(comparativo, ventas_mes, on='VDE', how='left').fillna(0)

            # Calcular variaciones
            comparativo['Variación $'] = comparativo['Ventas Mes'] - comparativo['Promedio Mensual']
            comparativo['Variación %'] = (comparativo['Ventas Mes'] / comparativo['Promedio Mensual'] - 1) * 100

            # Formatear y mostrar
            st.dataframe(
                comparativo.style.format({
                    'Total Año': "${:,.2f}",
                    'Promedio Mensual': "${:,.2f}",
                    'Ventas Mes': "${:,.2f}",
                    'Variación $': "${:,.2f}",
                    'Variación %': "{:.1f}%"
                }).applymap(
                    lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 
                            'color: red' if isinstance(x, (int, float)) and x < 0 else '', 
                    subset=['Variación $', 'Variación %']
                ),
                use_container_width=True
            )
            st.subheader("📊 Segmentación por Rango de Días sin Compra")

        df_segmentos = kpi_calculator.segmentar_inactividad_por_rango(current_df)

        if not df_segmentos.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("📋 **Tabla de Segmentos**")
                st.dataframe(
                    df_segmentos.style.format({
                        'Clientes en Rango': "{:,.0f}",
                        'Promedio Días Inactivos': "{:.1f}"
                    }).background_gradient(
                        subset=['Clientes en Rango'],
                        cmap='Oranges'
                    ),
                    use_container_width=True
                )

            with col2:
                st.markdown("📈 **Gráfico de Barras**")
                fig_bar = px.bar(
                    df_segmentos,
                    x='Rango de Días Sin Compra',
                    y='Clientes en Rango',
                    color='Clientes en Rango',
                    text='Clientes en Rango',
                    labels={'Clientes en Rango': 'Clientes'},
                    title='Clientes por Rango de Días Inactivos',
                    color_continuous_scale='Oranges'
                )
                fig_bar.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis_title="Rango de Días",
                    yaxis_title="Clientes",
                    height=350
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.info("No se encontraron clientes inactivos.")

            st.subheader("📂 Detalle de Clientes por Rango de Inactividad")

        # Rango disponibles desde los segmentos
        rango_opciones = [(1, 15), (16, 30), (31, 60), (61, 90), (91, 999)]
        rango_labels = [f"{r[0]} - {r[1]} días" for r in rango_opciones]
        rango_seleccionado_label = st.selectbox("Selecciona un rango para ver clientes", options=rango_labels)
        rango_seleccionado = rango_opciones[rango_labels.index(rango_seleccionado_label)]

        # Obtener clientes en ese rango
        clientes_rango_df = kpi_calculator.obtener_clientes_por_rango_inactivo(current_df, rango=rango_seleccionado)

        if not clientes_rango_df.empty:
            st.markdown(f"**Total clientes en este rango: {len(clientes_rango_df):,}**")
            st.dataframe(
                clientes_rango_df.style.format({'DIAS_INACTIVO': '{:.0f}'}),
                use_container_width=True
            )

            # Descargar como CSV
            csv = clientes_rango_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar listado (CSV)",
                data=csv,
                file_name=f"clientes_inactivos_{rango_seleccionado[0]}_{rango_seleccionado[1]}_dias.csv",
                mime='text/csv'
            )
        else:
            st.info("No hay clientes en este rango.")
        
    #     Mapa de calor de ventas por día de semana y hora
    #     st.markdown("**🔥 Mapa de Calor de Ventas por Día y Hora**")
        
    #     if 'HORA' in current_df.columns:
    #         current_df['HORA'] = pd.to_datetime(current_df['HORA'], format='%H:%M:%S').dt.hour
    #         current_df['DIA_SEMANA'] = current_df['FECHA'].dt.day_name()
            
    #         dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    #         current_df['DIA_SEMANA'] = pd.Categorical(current_df['DIA_SEMANA'], categories=dias_orden, ordered=True)
            
    #         heatmap_data = current_df.groupby(['DIA_SEMANA', 'HORA'])['MONTO'].sum().unstack().fillna(0)
            
    #         fig_heatmap = px.imshow(
    #             heatmap_data,
    #             labels=dict(x="Hora del Día", y="Día de la Semana", color="Ventas"),
    #             x=heatmap_data.columns,
    #             y=heatmap_data.index,
    #             aspect="auto",
    #             color_continuous_scale='Blues'
    #         )
    #         fig_heatmap.update_layout(
    #             plot_bgcolor='white',
    #             paper_bgcolor='white',
    #             xaxis=dict(showgrid=False),
    #             yaxis=dict(showgrid=False)
    #         )
    #         st.plotly_chart(fig_heatmap, use_container_width=True, key="mapa_calor_tab1")
    #     else:
    #         st.warning("No se encontró la columna 'HORA' en los datos")
    # else:
    #     st.warning("No hay datos con los filtros seleccionados")

def mostrar_cumplimiento_metas(kpi_calculator, current_df, year_filter, meses_seleccionados):
    st.header("🎯 Cumplimiento de Metas")
    
    if not current_df.empty and meses_seleccionados:
        mes_analisis = meses_seleccionados[0]
        cumplimiento = kpi_calculator.calcular_cumplimiento_metas(current_df, mes_analisis, year_filter)
        
        if cumplimiento:
            # Renombrar las columnas antes de mostrar el dataframe
            df_renombrado = cumplimiento['dataframe'].rename(columns={
                'MONTO_x': 'MONTO REAL',
                'MONTO_y': 'MONTO META',
                'CANTIDAD_x': 'CANTIDAD REAL',
                'CANTIDAD_y': 'CANTIDAD META',
                'VDE': 'VENDEDOR',
                'DOCUMENTO': 'FACTURAS',
                'CODIGO': 'CLIENTES',
                '% Cumplimiento Ventas': '% CUMPL. VENTAS',
                '% Cumplimiento Cajas': '% CUMPL. CANTIDAD'
            })
            
            st.dataframe(
                df_renombrado.style.format({
                    'MONTO REAL': "${:,.2f}",
                    'MONTO META': "${:,.2f}",
                    '% CUMPL. VENTAS': "{:.1f}%",
                    'CANTIDAD REAL': "{:,.0f}",
                    'CANTIDAD META': "{:,.0f}",
                    '% CUMPL. CANTIDAD': "{:.1f}%",
                    'CLIENTES': "{:,}",
                    'FACTURAS': "{:,}",
                    'Ticket Promedio': "${:,.2f}",
                    'Factura Promedio': "${:,.2f}"
                }).map(
                    lambda x: 'color: green' if isinstance(x, (int, float)) and x >= 100 else 
                            'color: orange' if isinstance(x, (int, float)) and x >= 70 else 
                            'color: red' if isinstance(x, (int, float)) and x < 70 else '',
                    subset=['% CUMPL. VENTAS', '% CUMPL. CANTIDAD']
                ).set_properties(
                    **{'background-color': '#f8f9fa'}, 
                    subset=pd.IndexSlice[df_renombrado.index[:-1], :]
                ).set_properties(
                    **{'background-color': '#3498db', 'color': 'white'}, 
                    subset=pd.IndexSlice[df_renombrado.index[-1], :]
                ),
                use_container_width=True
            )
            
            # Sección de Esfuerzo Diario Requerido
            st.subheader("📊 Esfuerzo Diario Requerido para Alcanzar Metas")

            resumen = cumplimiento['dataframe']
            required_cols = ['VDE', 'MONTO_x', 'MONTO_y', 'CANTIDAD_x', 'CANTIDAD_y', 'DOCUMENTO']
            if all(col in resumen.columns for col in required_cols):
                total_row = resumen.iloc[-1]
                hoy = datetime.now().date()
                primer_dia_mes = hoy.replace(day=1)
                ultimo_dia_mes = (primer_dia_mes + timedelta(days=32)).replace(day=1) - timedelta(days=1)

                def calcular_dias_laborables(start, end):
                    return sum(
                        0.5 if (start + timedelta(days=i)).weekday() == 5 else 
                        1 if (start + timedelta(days=i)).weekday() != 6 else 0
                        for i in range((end - start).days + 1)
                    )

                dias_transcurridos = calcular_dias_laborables(primer_dia_mes, hoy)
                dias_totales = calcular_dias_laborables(primer_dia_mes, ultimo_dia_mes)
                dias_faltantes = max(0, dias_totales - dias_transcurridos)
                clientes_meta_total = 25 * dias_totales

                if dias_faltantes > 0:
                    # METAS
                    meta_monto = total_row['MONTO_y']
                    meta_cantidad = total_row['CANTIDAD_y']
                    meta_clientes_dia = 25
                    meta_ticket = meta_monto / meta_cantidad if meta_cantidad > 0 else 0
                    meta_factura = meta_monto / clientes_meta_total if clientes_meta_total > 0 else 0

                    # REALES
                    real_monto = total_row['MONTO_x']
                    real_cantidad = total_row['CANTIDAD_x']
                    real_clientes = total_row['DOCUMENTO']
                    real_clientes_dia = real_clientes / dias_transcurridos if dias_transcurridos > 0 else 0
                    real_ticket = real_monto / real_cantidad if real_cantidad > 0 else 0
                    real_factura = real_monto / real_clientes if real_clientes > 0 else 0

                    # REQUERIMIENTO
                    ventas_faltantes = max(0, meta_monto - real_monto)
                    ventas_diarias_requeridas = ventas_faltantes / dias_faltantes

                    cantidad_faltantes = max(0, meta_cantidad - real_cantidad)
                    cantidad_diarias_requeridas = cantidad_faltantes / dias_faltantes

                    clientes_faltantes = max(0, clientes_meta_total - real_clientes)
                    clientes_diarios_requeridos = clientes_faltantes / dias_faltantes if dias_faltantes > 0 else 0

                    ticket_requerido = ventas_diarias_requeridas / cantidad_diarias_requeridas if cantidad_diarias_requeridas > 0 else 0
                    factura_requerida = ventas_diarias_requeridas / clientes_diarios_requeridos if clientes_diarios_requeridos > 0 else 0
                    
                    # PROYECCIÓN AL CIERRE
                    ventas_diarias_promedio = real_monto / dias_transcurridos if dias_transcurridos > 0 else 0
                    proyeccion_cierre = real_monto + (ventas_diarias_promedio * dias_faltantes)
                    porcentaje_proyeccion = (proyeccion_cierre / meta_monto) * 100 if meta_monto > 0 else 0

                    # TARJETAS
                    cols = st.columns(4)

                    with cols[0]:  # META
                        st.markdown(f"""
                        <div class="kpi-card" style="border-left-color: {COLORES['positivo']}">
                            <div class="kpi-title">🎯 Meta Mensual</div>
                            <div class="kpi-value">{format_monto(meta_monto)}</div>
                            <div class="kpi-subtext">
                            <div><span class="kpi-value">👥 Clientes/día: {meta_clientes_dia:.1f}</span></div>
                            <div><span class="kpi-value">📦 Cantidad/día: {format_cantidad(meta_cantidad/dias_totales)}</span></div>
                            <div><span class="kpi-value">🎫 Ticket: {format_monto(meta_ticket)}</span></div>
                            <div><span class="kpi-value">🧾 Factura: {format_monto(meta_factura)}</span></div>
                        </div>
                        """, unsafe_allow_html=True)

                    with cols[1]:  # REAL
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-title">📌 Real Acumulado</div>
                            <div class="kpi-value">{format_monto(real_monto)}</div>
                            <div class="kpi-subtext">
                                <div><span class="kpi-value">👥 Clientes/día: <span class="kpi-value">{real_clientes_dia:.1f}</span><br>
                                <div><span class="kpi-value">📦 Cantidad: <span class="kpi-value">{format_cantidad(real_cantidad)}</span><br>
                                <div><span class="kpi-value">🎫 Ticket: <span class="kpi-value">{format_monto(real_ticket)}</span><br>
                                <div><span class="kpi-value">🧾 Factura: <span class="kpi-value">{format_monto(real_factura)}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with cols[2]:  # ESFUERZO
                        st.markdown(f"""
                        <div class="kpi-card" style="border-left-color: {COLORES['advertencia']}">
                            <div class="kpi-title">⚡ Esfuerzo Diario</div>
                            <div class="kpi-value">{format_monto(ventas_diarias_requeridas)}</div>
                            <div class="kpi-subtext">
                                <div><span class="kpi-value">👥 Clientes/día: <span class="kpi-value">{clientes_diarios_requeridos:.1f}</span><br>
                                <div><span class="kpi-value">📦 Cantidad/día: <span class="kpi-value">{format_cantidad(cantidad_diarias_requeridas)}</span><br>
                                <div><span class="kpi-value">🎫 Ticket: <span class="kpi-value">{format_monto(ticket_requerido)}</span><br>
                                <div><span class="kpi-value">🧾 Factura: <span class="kpi-value">{format_monto(factura_requerida)}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with cols[3]:  # PROYECCIÓN
                        color_proyeccion = (COLORES['positivo'] if porcentaje_proyeccion >= 100 else 
                                          COLORES['advertencia'] if porcentaje_proyeccion >= 70 else 
                                          COLORES['negativo'])
                        st.markdown(f"""
                        <div class="kpi-card" style="border-left-color: {color_proyeccion}">
                            <div class="kpi-title">📈 Proyección Cierre</div>
                            <div class="kpi-value">{format_monto(proyeccion_cierre)}</div>
                            <div class="kpi-subtext">
                                <div><span class="kpi-value">Días transcurridos: <span class="kpi-value">{dias_transcurridos:.1f}/{dias_totales:.1f}</span><br>
                                <div><span class="kpi-value">Tendencia actual: <span class="kpi-value">{porcentaje_proyeccion:.1f}%</span><br>
                                <div><span class="kpi-value">Venta diaria: <span class="kpi-value">{format_monto(ventas_diarias_promedio)}</span><br>
                                <div><span class="kpi-value">Días faltantes: <span class="kpi-value">{dias_faltantes:.1f}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                else:
                    st.warning("El mes ha concluido. No quedan días laborables para planificar.")
            else:
                st.error(f"Faltan columnas requeridas. Disponibles: {', '.join(resumen.columns)}")

def mostrar_analisis_pareto(kpi_calculator, current_df):
    st.header("📊 Análisis 20/80 (Principio de Pareto)")
    
    if not current_df.empty:
        # Análisis por Categoría de Productos
        st.subheader("📦 Análisis por Categoría de Productos")
        
        # Usar el método de análisis Pareto de productos de la clase KPICalculator
        pareto_productos = kpi_calculator.analisis_pareto_productos(current_df)
        
        if pareto_productos:
            # Mostrar resultados
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(crear_card(
                    "Total Categorías", 
                    pareto_productos['total_categorias'], 
                    'cantidad'
                ), unsafe_allow_html=True)
            with col2:
                st.markdown(crear_card(
                    "Ventas Totales", 
                    pareto_productos['total_ventas'], 
                    'monto'
                ), unsafe_allow_html=True)
            with col3:
                st.markdown(crear_card(
                    "Unidades Totales", 
                    pareto_productos['total_unidades'], 
                    'cantidad'
                ), unsafe_allow_html=True)
            
            # Mostrar tabla de categorías
            st.dataframe(
                pareto_productos['dataframe'].style.format({
                    'Ventas Totales': "${:,.2f}",
                    'Unidades Vendidas': "{:,.0f}",
                    '% Acumulado Ventas': "{:.1f}%",
                    '% Acumulado Unidades': "{:.1f}%",
                    'Facturas': "{:,.0f}"
                }).background_gradient(subset=['Ventas Totales'], cmap='Blues'),
                use_container_width=True,
                height=500
            )
        
            # Gráfico de Pareto por categorías
            fig_categorias = px.bar(
                pareto_productos['dataframe'].head(20),
                x='Categoría',
                y='Ventas Totales',
                title='Top 20 Categorías por Ventas',
                labels={'Ventas Totales': 'Ventas ($)', 'Categoría': 'Categoría de Productos'},
                color='Ventas Totales',
                color_continuous_scale='Blues',
                hover_data=['Unidades Vendidas', 'Subcategoría', 'Facturas']
            )
            
            fig_categorias.add_scatter(
                x=pareto_productos['dataframe'].head(20)['Categoría'],
                y=pareto_productos['dataframe'].head(20)['% Acumulado Ventas'],
                mode='lines+markers',
                name='% Acumulado Ventas',
                yaxis='y2',
                line=dict(color='orange', width=2)
            )
            
            fig_categorias.update_layout(
                yaxis2=dict(
                    title='% Acumulado Ventas',
                    overlaying='y',
                    side='right',
                    range=[0, 100]
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=600,
                xaxis_tickangle=-45,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_categorias, use_container_width=True, key="pareto_categorias_tab3")
        
        # Análisis de Pareto para clientes
        st.subheader("🧑‍💼 Análisis 20/80 - Clientes")
        
        # Usar el método de análisis Pareto de clientes de la clase KPICalculator
        pareto_clientes = kpi_calculator.analisis_pareto_clientes(current_df)
        
        if pareto_clientes:
            st.markdown(f"""
            <div class="card">
                <h4>Principio de Pareto (80/20) - Clientes</h4>
                <p><strong>Clientes que generan el 80% de ventas:</strong> {pareto_clientes['clientes_top_count']} de {pareto_clientes['total_clientes']} ({pareto_clientes['clientes_top_count']/pareto_clientes['total_clientes']*100:.1f}%)</p>
                <p><strong>Ventas generadas:</strong> ${pareto_clientes['ventas_top']:,.2f} (80%)</p>
                <p><strong>Ventas promedio por cliente top:</strong> ${pareto_clientes['clientes_top']['MONTO'].mean():,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Gráfico de Pareto
            fig = px.line(
                pareto_clientes['dataframe'],
                x='% Acumulado Clientes',
                y='% Acumulado Ventas',
                title='Análisis 80/20 - Clientes vs Ventas',
                markers=True
            )
            fig.add_vline(x=20, line_dash="dash", line_color="red")
            fig.add_hline(y=80, line_dash="dash", line_color="red")
            fig.update_layout(
                annotations=[
                    dict(x=20, y=80, xref="x", yref="y", text="Punto 80/20", showarrow=True)
                ],
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True, key="pareto_clientes_tab3")
            
            # Tabla de clientes top
            st.subheader("🏆 Top Clientes (80% de ventas)")
            st.dataframe(
                pareto_clientes['clientes_top'].style.format({
                    'MONTO': "${:,.2f}",
                    'DOCUMENTO': "{:,.0f}",
                    '% Acumulado Ventas': "{:.1f}%",
                    '% Acumulado Clientes': "{:.1f}%"
                }).background_gradient(subset=['MONTO'], cmap='Blues'),
                use_container_width=True
            )
    else:
        st.warning("No hay datos con los filtros seleccionados")

def mostrar_evolucion_productos(kpi_calculator, current_df):
    st.header("🔄 Evolución de Productos")
    
    if not current_df.empty and 'SUBCATEGORIA' in current_df.columns:
        productos = current_df['SUBCATEGORIA'].unique()
        seleccionados = st.multiselect(
            "Seleccionar productos para analizar",
            options=productos,
            default=productos[:2] if len(productos) >= 2 else productos,
            key="productos_evolucion"
        )
        
        if seleccionados:
            evolucion = current_df[current_df['SUBCATEGORIA'].isin(seleccionados)].groupby(
                ['MES', 'MES_ORDEN', 'SUBCATEGORIA']
            ).agg({
                'MONTO': 'sum',
                'CANTIDAD': 'sum'
            }).reset_index().sort_values('MES_ORDEN')
            
            fig1 = px.line(
                evolucion,
                x='MES',
                y='MONTO',
                color='SUBCATEGORIA',
                title='Evolución de Ventas por Mes',
                markers=True
            )
            fig1.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            
            fig2 = px.line(
                evolucion,
                x='MES',
                y='CANTIDAD',
                color='SUBCATEGORIA',
                title='Evolución de Unidades Vendidas por Mes',
                markers=True
            )
            fig2.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)

def mostrar_comparativo_anual(kpi_calculator, current_df):
    st.header("📅 Comparativo Anual")
    
    # Obtener los datos COMPLETOS sin filtrar por año
    df_completo = kpi_calculator.ventas_df
    
    if not df_completo.empty:
        # Filtros especiales para esta pestaña
        col1, col2, col3 = st.columns(3)
        with col1:
            # Obtenemos TODOS los años disponibles, no solo el filtrado
            available_years = sorted(df_completo['YEAR'].unique(), reverse=True)
            
            if len(available_years) < 2:
                st.warning("Se necesitan datos de al menos 2 años para comparar")
                return
                
            años_comparar = st.multiselect(
                "Seleccionar 2 años a comparar",
                options=available_years,
                default=available_years[:2],
                max_selections=2
            )
        
        with col2:
            meses_comparar = st.multiselect(
                "Seleccionar meses a incluir",
                options=MESES_ORDEN,
                default=MESES_ORDEN
            )
            
        with col3:
            vendedor_options = ['Todos'] + sorted(df_completo['VDE'].dropna().unique().tolist())
            vendedor_filter = st.selectbox(
                "Filtrar por vendedor",
                options=vendedor_options,
                index=0
            )
        
        if len(años_comparar) == 2:
            try:
                # Convertir años a strings para evitar problemas con tipos numéricos
                año1, año2 = años_comparar[0], años_comparar[1]
                año1_str, año2_str = str(año1), str(año2)
                
                # Crear dataframe comparativo
                df_comparativo = df_completo[
                    (df_completo['YEAR'].isin(años_comparar)) & 
                    (df_completo['MES'].isin(meses_comparar))
                ].copy()
                
                if vendedor_filter != 'Todos':
                    df_comparativo = df_comparativo[df_comparativo['VDE'] == vendedor_filter].copy()
                
                # Verificación crítica de datos
                if df_comparativo.empty:
                    st.error("⚠️ No hay datos con los filtros seleccionados")
                    st.stop()
                
                # Verificar que hay datos para ambos años
                años_presentes = df_comparativo['YEAR'].unique()
                if len(años_presentes) < 2:
                    st.error(f"⚠️ Solo hay datos para el año {años_presentes[0]} con los filtros seleccionados")
                    st.stop()
                    
                # ---------------------------------------------------------------------------------
                # 1. Tarjetas Resumen Comparativo
                # ---------------------------------------------------------------------------------
                st.subheader("📊 KPIs Comparativos")
                
                # Calcular métricas por año
                kpis = df_comparativo.groupby('YEAR').agg({
                    'MONTO': ['sum', 'mean'],
                    'DOCUMENTO': ['nunique', 'count'],
                    'CLIENTE': 'nunique',
                    'CANTIDAD': 'sum'
                }).reset_index()
                
                # Renombrar columnas
                kpis.columns = [
                    'Año', 'Ventas Totales', 'Ticket Promedio', 
                    'Clientes Únicos', 'Transacciones', 'Facturas', 
                    'Unidades Vendidas'
                ]
                
                # Calcular variaciones
                kpis_año1 = kpis[kpis['Año'] == año1].iloc[0]
                kpis_año2 = kpis[kpis['Año'] == año2].iloc[0]
                
                variacion_ventas = (kpis_año1['Ventas Totales'] - kpis_año2['Ventas Totales']) / kpis_año2['Ventas Totales'] * 100
                variacion_clientes = (kpis_año1['Clientes Únicos'] - kpis_año2['Clientes Únicos']) / kpis_año2['Clientes Únicos'] * 100
                variacion_unidades = (kpis_año1['Unidades Vendidas'] - kpis_año2['Unidades Vendidas']) / kpis_año2['Unidades Vendidas'] * 100
                
                # Mostrar cards con formato consistente
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(crear_card(
                        f"Ventas {año1} vs {año2}",
                        variacion_ventas,
                        'porcentaje',
                        f"{format_monto(kpis_año1['Ventas Totales'])} vs {format_monto(kpis_año2['Ventas Totales'])}"
                    ), unsafe_allow_html=True)
                
                with col2:
                    st.markdown(crear_card(
                        f"Clientes {año1} vs {año2}",
                        variacion_clientes,
                        'porcentaje',
                        f"{format_cantidad(kpis_año1['Clientes Únicos'])} vs {format_cantidad(kpis_año2['Clientes Únicos'])}"
                    ), unsafe_allow_html=True)
                
                with col3:
                    st.markdown(crear_card(
                        f"Unidades {año1} vs {año2}",
                        variacion_unidades,
                        'porcentaje',
                        f"{format_cantidad(kpis_año1['Unidades Vendidas'])} vs {format_cantidad(kpis_año2['Unidades Vendidas'])}"
                    ), unsafe_allow_html=True)
                
                # ---------------------------------------------------------------------------------
                # 2 Tabla de KPIs Generales (con formato mejorado)
                # ---------------------------------------------------------------------------------
                st.subheader("📊 KPIs Generales")

                # Calcular métricas por año
                kpis = df_comparativo.groupby('YEAR').agg({
                    'MONTO': ['sum', 'mean'],
                    'DOCUMENTO': ['nunique', 'count'],
                    'CLIENTE': 'nunique',
                    'CANTIDAD': 'sum'
                }).reset_index()

                # Renombrar columnas
                kpis.columns = [
                    'Año', 'Ventas Totales', 'Promedio Monto', 
                    'Transacciones', 'Total Documentos', 'Clientes Únicos', 
                    'Unidades Vendidas'
                ]

                # Calcular KPIs correctamente
                kpis['Ticket Promedio'] = kpis['Ventas Totales'] / kpis['Transacciones']
                kpis['Factura Promedio'] = kpis['Ventas Totales'] / kpis['Clientes Únicos']

                # Reorganizar columnas finales (y eliminar 'Promedio Monto' si no se necesita)
                kpis = kpis[['Año', 'Ventas Totales', 'Ticket Promedio', 'Factura Promedio',
                            'Clientes Únicos', 'Transacciones', 'Unidades Vendidas']]

                # Pivotear para comparación
                kpis_pivot = kpis.set_index('Año').T

                # Calcular diferencias si hay 2 años
                if len(kpis_pivot.columns) == 2:
                    año1, año2 = sorted(kpis_pivot.columns)  # Ordenar para asegurar coherencia
                    kpis_pivot['Diferencia'] = kpis_pivot[año2] - kpis_pivot[año1]
                    kpis_pivot['Variación %'] = ((kpis_pivot[año2] / kpis_pivot[año1]) - 1) * 100

                # Función para formato
                def format_kpi_value(val, is_money=False, is_percent=False):
                    if pd.isna(val):
                        return "N/A"
                    
                    color = ""
                    if isinstance(val, (int, float)):
                        if is_percent:
                            color = "color: #2ecc71" if val >= 0 else "color: #e74c3c"
                            val_str = f"{val:+.1f}%"
                        else:
                            val_str = f"${val:,.2f}" if is_money else f"{val:,.0f}"
                    else:
                        val_str = str(val)
                    
                    return f'<span style="{color}">{val_str}</span>'

                # Crear HTML para tabla
                table_html = """
                <table class="kpi-table">
                    <thead>
                        <tr>
                            <th>KPI</th>
                """

                for col in kpis_pivot.columns:
                    table_html += f"<th>{col}</th>"

                table_html += "</tr></thead><tbody>"

                for index, row in kpis_pivot.iterrows():
                    table_html += f"<tr><td>{index}</td>"
                    
                    for col in kpis_pivot.columns:
                        is_money = index in ['Ventas Totales', 'Ticket Promedio', 'Factura Promedio', 'Ventas/Día']
                        is_percent = '%' in str(col) or 'Variación' in str(index)
                        val = row[col]
                        formatted_val = format_kpi_value(val, is_money, is_percent)
                        table_html += f"<td>{formatted_val}</td>"
                    
                    table_html += "</tr>"

                table_html += "</tbody></table>"

                st.markdown(table_html, unsafe_allow_html=True)

                # ---------------------------------------------------------------------------------
                # 3. Tabla Comparativa Detallada
                # ---------------------------------------------------------------------------------
                st.subheader("📋 Detalle Comparativo")
                
                # Calcular métricas por categoría y año
                ventas_categoria = df_comparativo.groupby(['YEAR', 'CATEGORIA']).agg({
                    'MONTO': 'sum',
                    'CANTIDAD': 'sum',
                    'DOCUMENTO': 'nunique'
                }).reset_index()
                
                # Pivotear para tener años como columnas
                ventas_pivot = ventas_categoria.pivot(
                    index='CATEGORIA',
                    columns='YEAR',
                    values=['MONTO', 'CANTIDAD', 'DOCUMENTO']
                ).reset_index()
                
                # Renombrar columnas para evitar MultiIndex
                ventas_pivot.columns = [
                    'Categoría',
                    f'Ventas_{año1_str}', f'Ventas_{año2_str}',
                    f'Unidades_{año1_str}', f'Unidades_{año2_str}',
                    f'Facturas_{año1_str}', f'Facturas_{año2_str}'
                ]
                
                # Calcular tickets promedio primero
                ventas_pivot[f'Ticket_Prom_{año1_str}'] = ventas_pivot[f'Ventas_{año1_str}'] / ventas_pivot[f'Facturas_{año1_str}']
                ventas_pivot[f'Ticket_Prom_{año2_str}'] = ventas_pivot[f'Ventas_{año2_str}'] / ventas_pivot[f'Facturas_{año2_str}']
                
                # Ahora calcular diferencias y variaciones
                ventas_pivot['Dif_Ventas'] = ventas_pivot[f'Ventas_{año1_str}'] - ventas_pivot[f'Ventas_{año2_str}']
                ventas_pivot['Var_Ventas_%'] = (ventas_pivot[f'Ventas_{año1_str}'] / ventas_pivot[f'Ventas_{año2_str}'] - 1) * 100
                
                ventas_pivot['Dif_Unidades'] = ventas_pivot[f'Unidades_{año1_str}'] - ventas_pivot[f'Unidades_{año2_str}']
                ventas_pivot['Var_Unidades_%'] = (ventas_pivot[f'Unidades_{año1_str}'] / ventas_pivot[f'Unidades_{año2_str}'] - 1) * 100
                
                ventas_pivot['Dif_Ticket'] = ventas_pivot[f'Ticket_Prom_{año1_str}'] - ventas_pivot[f'Ticket_Prom_{año2_str}']
                
                # Ordenar por variación de ventas
                ventas_pivot = ventas_pivot.sort_values('Var_Ventas_%', ascending=False)
                
                # Agregar fila de totales
                total_row = {
                    'Categoría': 'TOTAL',
                    f'Ventas_{año1_str}': ventas_pivot[f'Ventas_{año1_str}'].sum(),
                    f'Ventas_{año2_str}': ventas_pivot[f'Ventas_{año2_str}'].sum(),
                    'Dif_Ventas': ventas_pivot['Dif_Ventas'].sum(),
                    'Var_Ventas_%': (ventas_pivot[f'Ventas_{año1_str}'].sum() / ventas_pivot[f'Ventas_{año2_str}'].sum() - 1) * 100,
                    f'Unidades_{año1_str}': ventas_pivot[f'Unidades_{año1_str}'].sum(),
                    f'Unidades_{año2_str}': ventas_pivot[f'Unidades_{año2_str}'].sum(),
                    'Dif_Unidades': ventas_pivot['Dif_Unidades'].sum(),
                    'Var_Unidades_%': (ventas_pivot[f'Unidades_{año1_str}'].sum() / ventas_pivot[f'Unidades_{año2_str}'].sum() - 1) * 100,
                    f'Ticket_Prom_{año1_str}': ventas_pivot[f'Ventas_{año1_str}'].sum() / ventas_pivot[f'Facturas_{año1_str}'].sum(),
                    f'Ticket_Prom_{año2_str}': ventas_pivot[f'Ventas_{año2_str}'].sum() / ventas_pivot[f'Facturas_{año2_str}'].sum(),
                    'Dif_Ticket': (ventas_pivot[f'Ventas_{año1_str}'].sum() / ventas_pivot[f'Facturas_{año1_str}'].sum()) - 
                                  (ventas_pivot[f'Ventas_{año2_str}'].sum() / ventas_pivot[f'Facturas_{año2_str}'].sum())
                }
                
                ventas_final = pd.concat([ventas_pivot, pd.DataFrame([total_row])], ignore_index=True)
                
                # Mostrar tabla con formato consistente
                st.dataframe(
                    ventas_final.style.format({
                        f'Ventas_{año1_str}': "${:,.2f}",
                        f'Ventas_{año2_str}': "${:,.2f}",
                        'Dif_Ventas': "${:,.2f}",
                        'Var_Ventas_%': "{:+.1f}%",
                        f'Unidades_{año1_str}': "{:,.0f}",
                        f'Unidades_{año2_str}': "{:,.0f}",
                        'Dif_Unidades': "{:+,.0f}",
                        'Var_Unidades_%': "{:+.1f}%",
                        f'Ticket_Prom_{año1_str}': "${:,.2f}",
                        f'Ticket_Prom_{año2_str}': "${:,.2f}",
                        'Dif_Ticket': "${:+.2f}"
                    }).apply(
                        lambda x: ['color: green' if isinstance(v, (int, float)) and v > 0 
                                else 'color: red' if isinstance(v, (int, float)) and v < 0 
                                else '' for v in x],
                        subset=['Dif_Ventas', 'Var_Ventas_%', 'Dif_Unidades', 'Var_Unidades_%', 'Dif_Ticket'],
                        axis=1
                    ).set_properties(
                        **{'background-color': '#f8f9fa'}, 
                        subset=pd.IndexSlice[ventas_final.index[:-1], :]
                    ).set_properties(
                        **{'background-color': '#3498db', 'color': 'white'}, 
                        subset=pd.IndexSlice[ventas_final.index[-1], :]
                    ),
                    use_container_width=True,
                    height=600
                )
                
                # ---------------------------------------------------------------------------------
                # 4. Gráficos Comparativos
                # ---------------------------------------------------------------------------------
                st.subheader("📈 Visualización Comparativa")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gráfico de barras comparativo
                    fig_comparativo = px.bar(
                        ventas_categoria,
                        x='CATEGORIA',
                        y='MONTO',
                        color='YEAR',
                        barmode='group',
                        title=f'Comparativo de Ventas por Categoría',
                        labels={'MONTO': 'Ventas ($)', 'CATEGORIA': 'Categoría', 'YEAR': 'Año'},
                        height=400
                    )
                    fig_comparativo.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=False)
                    )
                    st.plotly_chart(fig_comparativo, use_container_width=True)
                
                with col2:
                    # Gráfico de evolución mensual comparativa - Versión corregida
                    evolucion_mensual = df_comparativo.groupby(['YEAR', 'MES', 'MES_ORDEN'])['MONTO'].sum().reset_index()

                    # Crear un diccionario de mapeo de meses a su orden numérico
                    meses_orden = {mes: i+1 for i, mes in enumerate(MESES_ORDEN)}

                    # Asegurar que el DataFrame tenga la columna de orden de mes
                    evolucion_mensual['MES_ORDEN'] = evolucion_mensual['MES'].map(meses_orden)

                    # Ordenar los datos por año y luego por orden de mes
                    evolucion_mensual = evolucion_mensual.sort_values(['YEAR', 'MES_ORDEN'])

                    # Convertir la columna 'MES' a tipo categórico con el orden correcto
                    evolucion_mensual['MES'] = pd.Categorical(
                        evolucion_mensual['MES'],
                        categories=MESES_ORDEN,
                        ordered=True
                    )

                    fig_evolucion = px.line(
                        evolucion_mensual,
                        x='MES',
                        y='MONTO',
                        color='YEAR',
                        title='Evolución Mensual Comparativa',
                        markers=True,
                        labels={'MONTO': 'Ventas ($)', 'MES': 'Mes', 'YEAR': 'Año'},
                        height=400
                    )
                    fig_evolucion.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        xaxis=dict(
                            showgrid=False,
                            type='category',
                            categoryorder='array',
                            categoryarray=MESES_ORDEN
                        ),
                        yaxis=dict(showgrid=False)
                    )
                    st.plotly_chart(fig_evolucion, use_container_width=True)
                
                # ---------------------------------------------------------------------------------
                # 5. Análisis de Productos
                # ---------------------------------------------------------------------------------
                st.subheader("📦 Análisis Comparativo de Productos")
                
                try:
                    # Obtener lista de productos por año (versión corregida)
                    productos_año1 = df_comparativo[df_comparativo['YEAR'] == año1]['SUBCATEGORIA'].dropna().unique()
                    productos_año2 = df_comparativo[df_comparativo['YEAR'] == año2]['SUBCATEGORIA'].dropna().unique()
                    
                    # Convertir a conjuntos de strings
                    productos_año1 = set(str(p) for p in productos_año1 if pd.notna(p))
                    productos_año2 = set(str(p) for p in productos_año2 if pd.notna(p))
                    
                    if len(productos_año1) == 0 or len(productos_año2) == 0:
                        st.warning(f"No hay datos de productos para {'ambos años' if len(productos_año1) == 0 and len(productos_año2) == 0 else f'año {año1}' if len(productos_año1) == 0 else f'año {año2}'}")
                    else:
                        # Crear DataFrame comparativo de productos
                        productos_comp = df_comparativo.groupby(['YEAR', 'SUBCATEGORIA'])['MONTO'].sum().unstack(level=0).reset_index()
                        
                        # Renombrar columnas de años
                        productos_comp.columns = ['Producto', f'Ventas_{año1}', f'Ventas_{año2}']
                        
                        # Filtrar solo productos presentes en ambos años para una comparación válida
                        productos_comunes = productos_año1 & productos_año2  # Intersección de conjuntos
                        productos_comp = productos_comp[productos_comp['Producto'].isin(productos_comunes)]
                        
                        if len(productos_comp) == 0:
                            st.warning(f"No hay productos comunes entre {año1} y {año2} para comparar")
                        else:
                            # Calcular variaciones
                            productos_comp['Crecimiento'] = (productos_comp[f'Ventas_{año1}'] / productos_comp[f'Ventas_{año2}'].replace(0, np.nan) - 1) * 100
                            productos_comp['Diferencia'] = productos_comp[f'Ventas_{año1}'] - productos_comp[f'Ventas_{año2}']
                            
                            # Mostrar top 10 productos con mayor crecimiento
                            st.markdown(f"**🚀 Top 10 Productos con Mayor Crecimiento ({año1} vs {año2})**")
                            top_crecimiento = productos_comp.sort_values('Crecimiento', ascending=False).head(10)
                            
                            st.dataframe(
                                top_crecimiento.style.format({
                                    f'Ventas_{año1}': "${:,.2f}",
                                    f'Ventas_{año2}': "${:,.2f}",
                                    'Diferencia': "${:,.2f}",
                                    'Crecimiento': "{:+.1f}%"
                                }).apply(
                                    lambda x: ['color: green' if isinstance(v, (int, float)) and v > 0 
                                            else 'color: red' if isinstance(v, (int, float)) and v < 0 
                                            else '' for v in x],
                                    subset=['Diferencia', 'Crecimiento'],
                                    axis=1
                                ).background_gradient(
                                    cmap='RdYlGn',
                                    subset=['Crecimiento']
                                ),
                                use_container_width=True
                            )
                            
                            # Mostrar top 10 productos con mayor decrecimiento
                            st.markdown(f"**⚠️ Top 10 Productos con Mayor Decrecimiento ({año1} vs {año2})**")
                            top_decrecimiento = productos_comp.sort_values('Crecimiento').head(10)
                            
                            st.dataframe(
                                top_decrecimiento.style.format({
                                    f'Ventas_{año1}': "${:,.2f}",
                                    f'Ventas_{año2}': "${:,.2f}",
                                    'Diferencia': "${:,.2f}",
                                    'Crecimiento': "{:+.1f}%"
                                }).apply(
                                    lambda x: ['color: green' if isinstance(v, (int, float)) and v > 0 
                                            else 'color: red' if isinstance(v, (int, float)) and v < 0 
                                            else '' for v in x],
                                    subset=['Diferencia', 'Crecimiento'],
                                    axis=1
                                ).background_gradient(
                                    cmap='RdYlGn_r',
                                    subset=['Crecimiento']
                                ),
                                use_container_width=True
                            )
                            
                            # Mostrar productos nuevos y desaparecidos
                            nuevos_productos = productos_año1 - productos_año2
                            productos_desaparecidos = productos_año2 - productos_año1
                            
                            if nuevos_productos:
                                st.markdown(f"**🆕 Productos nuevos en {año1}**")
                                ventas_nuevos = df_comparativo[
                                    (df_comparativo['YEAR'] == año1) & 
                                    (df_comparativo['SUBCATEGORIA'].isin(nuevos_productos))
                                ].groupby('SUBCATEGORIA')['MONTO'].sum().reset_index()
                                st.dataframe(
                                    ventas_nuevos.sort_values('MONTO', ascending=False).style.format({
                                        'MONTO': "${:,.2f}"
                                    }),
                                    use_container_width=True
                                )
                            
                            if productos_desaparecidos:
                                st.markdown(f"**❌ Productos que dejaron de venderse en {año1}**")
                                ventas_desaparecidos = df_comparativo[
                                    (df_comparativo['YEAR'] == año2) & 
                                    (df_comparativo['SUBCATEGORIA'].isin(productos_desaparecidos))
                                ].groupby('SUBCATEGORIA')['MONTO'].sum().reset_index()
                                st.dataframe(
                                    ventas_desaparecidos.sort_values('MONTO', ascending=False).style.format({
                                        'MONTO': "${:,.2f}"
                                    }),
                                    use_container_width=True
                                )
                except Exception as e:
                    st.error(f"Error al analizar productos: {str(e)}")
                    st.write("Detalle del error:", traceback.format_exc())
                
            except Exception as e:
                st.error(f"Error inesperado: {str(e)}")
                st.write("Detalle del error:", traceback.format_exc())
        else:
            st.warning("Por favor selecciona exactamente 2 años para comparar")
    else:
        st.warning("No hay datos disponibles para realizar la comparación")

def mostrar_segmentacion_clientes(kpi_calculator, current_df):
    st.header("📊 Segmentación Avanzada de Clientes")
    
    if not current_df.empty:
        # Usar el método de segmentación de la clase KPICalculator
        segmentacion = kpi_calculator.segmentar_clientes(current_df)
        
        if not segmentacion.empty:
            # Asegurarnos de que tenemos la columna del vendedor
            if 'VDE' in current_df.columns and 'VDE' not in segmentacion.columns:
                # Agregar el nombre del vendedor a la segmentación
                vendedores = current_df[['CLIENTE', 'VDE']].drop_duplicates()
                segmentacion = segmentacion.merge(vendedores, on='CLIENTE', how='left')
                    
            # Resumen por segmento
            resumen_segmentos = segmentacion.groupby('Segmento').agg({
                'CLIENTE': 'count',
                'Monto': 'sum'
            }).rename(columns={
                'CLIENTE': 'CantidadClientes',
                'Monto': 'VentasTotales'
            }).sort_values('VentasTotales', ascending=False)
            
        
            # Calcular porcentajes
            resumen_segmentos['%Clientes'] = (resumen_segmentos['CantidadClientes'] / resumen_segmentos['CantidadClientes'].sum()) * 100
            resumen_segmentos['%Ventas'] = (resumen_segmentos['VentasTotales'] / resumen_segmentos['VentasTotales'].sum()) * 100
            
            # Mostrar cards con resumen por segmento
            st.subheader("🧩 Resumen por Segmento")
            cols = st.columns(5)
            
            for i, (segmento, datos) in enumerate(resumen_segmentos.iterrows()):
                with cols[i % 5]:
                    st.markdown(crear_card(
                        segmento,
                        datos['CantidadClientes'],
                        'cantidad',
                        f"Clientes ({datos['%Clientes']:.1f}%) - Ventas: {format_monto(datos['VentasTotales'])} ({datos['%Ventas']:.1f}%)"
                    ), unsafe_allow_html=True)
            
            # Gráficos de distribución
            col1, col2 = st.columns(2)
            with col1:
                fig_pie = px.pie(
                    resumen_segmentos.reset_index(),
                    names='Segmento',
                    values='VentasTotales',
                    title='Distribución de Ventas por Segmento',
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                fig_pie.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_bar = px.bar(
                    resumen_segmentos.reset_index(),
                    x='Segmento',
                    y='CantidadClientes',
                    title='Clientes por Segmento',
                    color='Segmento',
                    text='CantidadClientes',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_bar.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Verificar si tenemos datos de vendedores
            if 'VDE' in segmentacion.columns:
                # Tablas por vendedor
                st.subheader("📊 Distribución por Vendedor")
                
                # Tabla 1: Cantidad de clientes por vendedor y segmento
                st.markdown("**Cantidad de Clientes por Vendedor y Segmento**")
                vendedor_cantidades = segmentacion.groupby(['VDE', 'Segmento'])['CLIENTE'].count().unstack().fillna(0)
                
                # Ordenar columnas según los segmentos disponibles
                segmentos_ordenados = resumen_segmentos.index.tolist()
                vendedor_cantidades = vendedor_cantidades[segmentos_ordenados]
                
                # Calcular totales
                vendedor_cantidades['Total'] = vendedor_cantidades.sum(axis=1)
                vendedor_cantidades.loc['Total'] = vendedor_cantidades.sum()
                
                st.dataframe(
                    vendedor_cantidades.style.format("{:,.0f}").apply(
                        lambda x: ['background-color: #f8f9fa' if x.name == 'Total' else '' for _ in x],
                        axis=1
                    ).background_gradient(
                        cmap='Blues',
                        subset=segmentos_ordenados
                    ),
                    use_container_width=True
                )
                
                # Tabla 2: Monto de ventas por vendedor y segmento
                st.markdown("**Ventas por Vendedor y Segmento**")
                vendedor_montos = segmentacion.groupby(['VDE', 'Segmento'])['Monto'].sum().unstack().fillna(0)
                
                # Ordenar columnas según los segmentos disponibles
                vendedor_montos = vendedor_montos[segmentos_ordenados]
                
                # Calcular totales
                vendedor_montos['Total'] = vendedor_montos.sum(axis=1)
                vendedor_montos.loc['Total'] = vendedor_montos.sum()
                
                st.dataframe(
                    vendedor_montos.style.format("${:,.2f}").apply(
                        lambda x: ['background-color: #f8f9fa' if x.name == 'Total' else '' for _ in x],
                        axis=1
                    ).background_gradient(
                        cmap='Greens',
                        subset=segmentos_ordenados
                    ),
                    use_container_width=True
                )
            else:
                st.warning("No se encontró información de vendedores en los datos segmentados")
            
            # Tabla detallada de clientes
            st.subheader("🧑‍💼 Detalle de Clientes por Segmento")
            
            col1, col2 = st.columns(2)
            with col1:
                segmento_seleccionado = st.selectbox(
                    "Filtrar por segmento:",
                    options=resumen_segmentos.index.tolist(),
                    index=0,
                    key="segmento_select"
                )
            
            with col2:
                vendedor_options = ['Todos'] + segmentacion['VDE'].dropna().unique().tolist() if 'VDE' in segmentacion.columns else ['Todos']
                vendedor_seleccionado = st.selectbox(
                    "Filtrar por vendedor:",
                    options=vendedor_options,
                    index=0,
                    key="vendedor_select"
                )
            
            # Aplicar filtros
            clientes_filtrados = segmentacion[segmentacion['Segmento'] == segmento_seleccionado]
            
            if vendedor_seleccionado != 'Todos' and 'VDE' in segmentacion.columns:
                clientes_filtrados = clientes_filtrados[clientes_filtrados['VDE'] == vendedor_seleccionado]
            
            # Mostrar tabla
            st.dataframe(
                clientes_filtrados.sort_values('Monto', ascending=False).style.format({
                    'Recencia': "{:,.0f}",
                    'Frecuencia': "{:,.0f}",
                    'Monto': "${:,.2f}"
                }).bar(
                    subset='Monto',
                    color='#5fba7d'
                ),
                use_container_width=True,
                height=400
            )
            
            # Exportar datos
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📤 Exportar datos de segmentación", key="export_all"):
                    csv = segmentacion.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar CSV completo",
                        data=csv,
                        file_name=f"segmentacion_clientes_{datetime.now().year}.csv",
                        mime="text/csv",
                        key="download_all"
                    )
            
            with col2:
                if st.button("📤 Exportar datos filtrados", key="export_filtered"):
                    csv = clientes_filtrados.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar CSV filtrado",
                        data=csv,
                        file_name=f"clientes_{segmento_seleccionado}_{vendedor_seleccionado}.csv",
                        mime="text/csv",
                        key="download_filtered"
                    )

def mostrar_promociones():
    st.header("🔥 Estrategias de Promoción")
    
    # Segmentos específicos para el mercado dental
    segmentos_promos = {
        'Muy Frecuentes (12+)': {
            'descripcion': 'Clientes que compran frecuentemente y en grandes cantidades',
            'promociones': [
                "Programa de fidelización con puntos canjeables por productos premium",
                "10% de descuento en compras mayores a RD$50,000",
                "Acceso prioritario a nuevos productos y lanzamientos"
            ]
        },
        'Regulares (7-12)': {
            'descripcion': 'Clientes que compran regularmente pero no en grandes volúmenes',
            'promociones': [
                "5% de descuento en pedidos recurrentes",
                "Envío gratis en compras mayores a RD$30,000",
                "Kit de muestras gratis con cada compra"
            ]
        },
        'Frecuentes (4-6)': {
            'descripcion': 'Clientes con potencial de crecimiento en compras',
            'promociones': [
                "Oferta especial: 15% de descuento en primer pedido mayor a RD$20,000",
                "Asesoría gratuita con nuestro especialista",
                "Promoción combo: Compra 2 lleva 1 producto básico gratis"
            ]
        },
        'Esporádicos (2-3)': {
            'descripcion': 'Clientes que han comprado recientemente por primera vez',
            'promociones': [
                "Bienvenida: 20% de descuento en primera compra",
                "Guía de productos esenciales",
                "Sesión de capacitación sobre uso de productos"
            ]
        },
        'Ocasionales (1)': {
            'descripcion': 'Clientes que solían comprar frecuentemente pero han disminuido',
            'promociones': [
                "Oferta de reactivación: 25% de descuento en su próxima compra",
                "Encuesta personalizada para entender sus necesidades actuales",
                "Visita de nuestro representante con muestras gratis"
            ]
        },
        'Dormidos': {
            'descripcion': 'Clientes inactivos por más de 6 meses',
            'promociones': [
                "Campaña especial: 30% de descuento para su regreso",
                "Actualización de nuestro catálogo de productos",
                "Llamada personalizada de nuestro equipo comercial"
            ]
        }
    }
    
    # Mostrar promociones por segmento
    for segmento, info in segmentos_promos.items():
        with st.expander(f"🎯 Promociones para {segmento}"):
            st.markdown(f"**{info['descripcion']}**")
            for promo in info['promociones']:
                st.markdown(f"- {promo}")

def mostrar_manual_usuario(ventas_df):
    st.header("📚 Manual de Usuario")
    
    st.markdown("""
    <div class="card">
        <h2>Guía para el uso del Dashboard de Análisis de Ventas</h2>
        
        <h3>Filtros Globales</h3>
        <ul>
            <li><strong>Año</strong>: Selecciona el año a analizar</li>
            <li><strong>Vendedor</strong>: Filtra por vendedor (selecciona "Todos" para ver todos)</li>
            <li><strong>Meses</strong>: Permite seleccionar uno o varios meses para el análisis</li>
        </ul>
        
        <h3>Pestañas Disponibles</h3>
        
        <h4>1. 📈 Resumen Ventas</h4>
        <ul>
            <li>Muestra KPIs principales: Ventas totales, unidades vendidas, clientes únicos y ticket promedio</li>
            <li>Gráfico de ventas mensuales</li>
            <li>Comparativo interanual</li>
            <li>Proyecciones semanales y mensuales</li>
        </ul>
        
        <h4>2. 🎯 Cumplimiento Metas</h4>
        <ul>
            <li>Compara ventas reales vs metas establecidas</li>
            <li>Muestra porcentaje de cumplimiento para ventas y cantidades</li>
            <li>Incluye ticket promedio y factura promedio</li>
        </ul>
        
        <h4>3. 📊 Análisis 20/80</h4>
        <ul>
            <li>Identifica las categorías y clientes que generan el 80% de las ventas (Principio de Pareto)</li>
            <li>Muestra distribución de ventas por categoría y subcategoría</li>
        </ul>
        
        <h4>4. 🔄 Evolución Productos</h4>
        <ul>
            <li>Muestra tendencia de ventas para productos seleccionados</li>
            <li>Permite comparar múltiples productos</li>
        </ul>
        
        <h4>5. 📅 Comparativo Anual</h4>
        <ul>
            <li>Compara el desempeño entre dos años seleccionados</li>
            <li>Muestra variaciones porcentuales en ventas, clientes y unidades</li>
        </ul>
        
        <h4>6. 📊 Segmentación Clientes</h4>
        <ul>
            <li>Segmentación RFM (Recency, Frequency, Monetary)</li>
            <li>Análisis de migración entre segmentos</li>
            <li>Identificación de clientes en riesgo y oportunidades</li>
        </ul>
        
        <h4>7. 🔥 Promociones</h4>
        <ul>
            <li>Estrategias de promoción por segmento de cliente</li>
            <li>Generador de promociones personalizadas</li>
        </ul>
        
        <h4>8. 🔍 Análisis General</h4>
        <ul>
            <li>Vista consolidada con los KPIs más importantes</li>
            <li>Rankings de vendedores y productos</li>
            <li>Alertas gerenciales y recomendaciones automáticas</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# INTERFAZ PRINCIPAL
# =============================================

def main():
    # Cargar datos
    ventas_df, presupuesto_df, clientes_df = load_data()
    
    # Inicializar calculadora de KPIs
    kpi_calculator = KPICalculator(ventas_df, presupuesto_df, clientes_df)
    
    # Configurar filtros globales
    with st.sidebar:
        st.title("🔍 Filtros Globales")
        year_options = sorted(ventas_df['YEAR'].unique(), reverse=True)
        year_filter = st.selectbox("📅 Seleccionar Año", options=year_options, index=0)
        
        vendedor_options = ['Todos'] + sorted(ventas_df[ventas_df['YEAR'] == year_filter]['VDE'].dropna().unique().tolist())
        vendedor_filter = st.selectbox("👤 Seleccionar Vendedor", options=vendedor_options, index=0)
        
        semana_actual = datetime.now().isocalendar()[1]
        semana_filter = st.slider("🗓️ Semana (para pestañas 1-2)", 1, 52, semana_actual)
        
        meses_options = MESES_ORDEN
        meses_seleccionados = st.multiselect("🗓️ Seleccionar Meses", options=meses_options, default=meses_options)

    # Aplicar filtros globales
    current_df = kpi_calculator.aplicar_filtros(year_filter, meses_seleccionados, vendedor_filter)
    
    # Crear pestañas
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Resumen Ventas", "🎯 Cumplimiento Metas", "📊 Análisis 20/80", 
        "🔄 Evolución Productos", "📅 Comparativo Anual", "📊 Segmentación Clientes",
        "🔥 Promociones", "🔍 Análisis General", "📚 Manual de Usuario"
    ])
    
    # Cada pestaña ahora usa los métodos centralizados de KPICalculator
    with tab1:
        with st.expander("ℹ️ ¿Qué contiene este resumen?"):
            st.markdown("""
            Este panel presenta un resumen ejecutivo y visual del desempeño comercial:

            - **Ventas por Mes**: Gráfico de barras con evolución mensual.
            - **Proyección Semanal y Mensual**: Estimaciones de cierre según avance del tiempo y ventas acumuladas.
            - **KPIs Básicos**: Clientes, transacciones, ticket/factura promedio, frecuencia de compra.
            - **Detalle por Vendedor y Mes**: Incluye ventas y KPIs por cada vendedor.
            - **Comparativo de Desempeño**: Evalúa si los vendedores están por encima o por debajo de su promedio mensual.
            - **Segmentación por Días Inactivos**: Clientes agrupados por tiempo sin comprar.
            - **Clientes por Rango de Inactividad**: Detalle descargable para acción comercial.

            **Memoria de cálculo:**
            - Ticket Promedio = Ventas Totales / Transacciones.
            - Factura Promedio = Ventas Totales / Clientes Únicos.
            - Frecuencia de Compra = Promedio de facturas por cliente.
            - Proyección = Venta diaria × días laborales restantes.
            """)
        mostrar_resumen_ventas(kpi_calculator, current_df, year_filter, semana_filter, meses_seleccionados)
    
    with tab2:
        with st.expander("ℹ️ ¿Qué contiene este resumen?"):
            st.markdown("""
            Este panel presenta un resumen ejecutivo y visual del desempeño comercial:

            - **Ventas por Mes**: Gráfico de barras con evolución mensual.
            - **Proyección Semanal y Mensual**: Estimaciones de cierre según avance del tiempo y ventas acumuladas.
            - **KPIs Básicos**: Clientes, transacciones, ticket/factura promedio, frecuencia de compra.
            - **Detalle por Vendedor y Mes**: Incluye ventas y KPIs por cada vendedor.
            - **Comparativo de Desempeño**: Evalúa si los vendedores están por encima o por debajo de su promedio mensual.
            - **Segmentación por Días Inactivos**: Clientes agrupados por tiempo sin comprar.
            - **Clientes por Rango de Inactividad**: Detalle descargable para acción comercial.

            **Memoria de cálculo:**
            - Ticket Promedio = Ventas Totales / Transacciones.
            - Factura Promedio = Ventas Totales / Clientes Únicos.
            - Frecuencia de Compra = Promedio de facturas por cliente.
            - Proyección = Venta diaria × días laborales restantes.
            """)
        mostrar_cumplimiento_metas(kpi_calculator, current_df, year_filter, meses_seleccionados)
    
    with tab3:
        with st.expander("ℹ️ ¿Qué se analiza aquí?"):
            st.markdown("""
            Este módulo muestra alertas de negocio para actuar rápidamente:

            - **Vendedores bajo rendimiento**: Aquellos con menos del 70% de cumplimiento de su meta mensual.
            - **Clientes inactivos**: Segmentados por tiempo sin comprar (30, 60, 90+ días).
            - **Productos en caída**: Productos cuya venta ha disminuido más de un 30% respecto al mes anterior.

            **Memoria de cálculo:**
            - Inactividad: Se calcula la última fecha de compra por cliente.
            - Cumplimiento: Se compara el monto vendido con el presupuesto del mes.
            - Caída de productos: Se compara venta actual vs mes anterior, calculando % de variación.
    """)
        mostrar_analisis_pareto(kpi_calculator, current_df)
    
    with tab4:
        with st.expander("ℹ️ ¿Qué representa esta segmentación?"):
            st.markdown("""
            Los clientes son agrupados por su frecuencia de compra en el periodo seleccionado:

            - **Muy Frecuentes (12+)**: Más de 12 compras.
            - **Regulares (7-12)**: Compran cada mes.
            - **Frecuentes (4-6)**: Compran bimestral o trimestralmente.
            - **Esporádicos (2-3)**: Compras ocasionales.
            - **Ocasionales (1)**: Solo 1 compra en el periodo.

            Se muestra su producto principal y, si está disponible, su nombre y ubicación.

            **Memoria de cálculo:**
            - Se cuenta el número de documentos únicos por cliente.
            - Los rangos están definidos por intervalos de frecuencia.
            """)
        mostrar_evolucion_productos(kpi_calculator, current_df)
    
    with tab5:
        with st.expander("ℹ️ ¿Qué muestra este comparativo anual?"):
            st.markdown("""
            Compara el desempeño de ventas entre dos años seleccionados en base a los meses definidos:

            - **KPIs Comparados**: Ventas Totales, Ticket Promedio, Factura Promedio, Clientes Únicos, Transacciones y Unidades Vendidas.
            - **Variaciones Absolutas y Porcentuales** entre ambos años.
            - **Gráficos de barras y análisis visual del cambio**.

            **Memoria de cálculo:**
            - Ticket Promedio = Ventas Totales / Transacciones.
            - Factura Promedio = Ventas Totales / Clientes Únicos.
            - Variación % = (Año más reciente - Año anterior) / Año anterior × 100.
            """)
        mostrar_comparativo_anual(kpi_calculator, current_df)
    
    with tab6:
        with st.expander("📘 ¿Qué significa cada segmento de clientes?"):
            st.markdown("""
            - **Muy Frecuentes (12+)**: Clientes que han realizado más de 12 compras en el periodo analizado. Representan la base más leal y activa.
            - **Regulares (7-12)**: Clientes que compran mensualmente o casi todos los meses. Mantienen un hábito consistente de compra.
            - **Frecuentes (4-6)**: Clientes que realizan compras cada dos o tres meses. Son importantes pero menos constantes.
            - **Esporádicos (2-3)**: Clientes con baja frecuencia de compra. Necesitan estímulos para aumentar su recurrencia.
            - **Ocasionales (1)**: Clientes que solo han comprado una vez en el periodo. Es clave entender qué los motivó y por qué no han repetido.
            - **Dormidos**: Clientes que no han comprado en más de 6 meses. Representan una oportunidad de reactivación con estrategias adecuadas.
                Se muestra su producto principal y, si está disponible, su nombre y ubicación.
            **Memoria de cálculo:**
            - Se cuenta el número de documentos únicos por cliente.
            - Los rangos están definidos por intervalos de frecuencia.
            """)
        mostrar_segmentacion_clientes(kpi_calculator, current_df)
        
    
    with tab7:
        mostrar_promociones()
    
    with tab8:
        with st.expander("ℹ️ ¿Qué contiene este análisis?"):
            st.markdown("""
            Esta pestaña muestra un análisis integral del mes seleccionado, incluyendo:

            - **KPIs Globales**: Ventas Totales, Ticket Promedio, Factura Promedio, Clientes, Transacciones y Unidades Vendidas.
            - **Comparativos**: Variación respecto al mes anterior y al mismo mes del año anterior.
            - **Ranking de Vendedores**: Top 5 y Bottom 5 según ventas.
            - **Productos más y menos vendidos**, productos estancados y tendencias de crecimiento o caída.
            - **Mapa Comercial**: Distribución de clientes geolocalizados.
            - **Análisis 80/20 (Pareto)**: Clientes y productos que generan el mayor impacto.
            - **Proyecciones mensuales y semanales**: Según días laborables transcurridos y ventas acumuladas.
            - **Alertas Gerenciales** y **Recomendaciones Automáticas** basadas en desempeño.
            
            **Memoria de cálculo:**
            - Ticket Promedio = Ventas Totales / Transacciones.
            - Factura Promedio = Ventas Totales / Clientes Únicos.
            - Comparativos = (Actual - Anterior) / Anterior * 100.
            - Proyecciones = Ventas actuales / Días laborales transcurridos × Días laborales totales.
            """)
        mostrar_analisis_general(kpi_calculator, current_df, year_filter, meses_seleccionados)
    
    with tab9:
        mostrar_manual_usuario(ventas_df)

# =============================================
# ESTILOS CSS Y EJECUCIÓN
# =============================================

if __name__ == "__main__":
    # Estilos CSS personalizados
    st.markdown(f"""
    <style>
        /* Estilos base para cards */
        .segment-card, .kpi-card {{
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid {COLORES['neutro']};
        }}
        
        /* Segment Card específico */
        .segment-card h4 {{
            margin-top: 0;
            color: {COLORES['texto']};
            font-size: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .segment-card .value {{
            font-size: 24px;
            font-weight: bold;
            margin: 8px 0;
        }}
        
        .segment-card .subtext {{
            font-size: 12px;
            color: #7f8c8d;
        }}
        
        /* KPI Card específico */
        .kpi-card .kpi-title {{
            font-size: 14px;
            color: {COLORES['texto']};
            margin-bottom: 5px;
            font-weight: 600;
        }}
        
        .kpi-card .kpi-value {{
            font-size: 24px;
            font-weight: bold;
            color: {COLORES['texto']};
            margin: 8px 0;
        }}
        
        .kpi-card .kpi-subtext {{
            font-size: 12px;
            color: #7f8c8d;
            line-height: 1.4;
        }}
        
        .kpi-card .kpi-subtext div {{
            margin-top: 4px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        /* Estilos para tablas */
        .dataframe th {{
            background-color: {COLORES['fondo']} !important;
            font-weight: bold !important;
            text-align: center !important;
        }}
        
        .dataframe tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        
        .dataframe tr:hover {{
            background-color: #f1f1f1;
        }}
        
        /* Cards de análisis */
        .card {{
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .card h4 {{
            color: {COLORES['texto']};
            margin-top: 0;
        }}
        
        /* Mejoras generales de Streamlit */
        .stProgress > div > div > div > div {{
            background-color: {COLORES['positivo']};
        }}
        
        .st-bb, .st-at {{
            background-color: {COLORES['fondo']};
        }}
        
        .css-1vq4p4l {{
            padding: 1rem;
        }}
        
        /* Tablas personalizadas */
        .kpi-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .kpi-table th {{
            background-color: #f8f9fa;
            font-weight: bold;
            text-align: center;
            padding: 8px;
            border: 1px solid #ddd;
            font-size: 16px;
        }}
        .kpi-table td {{
            padding: 8px;
            border: 1px solid #ddd;
            text-align: right;
            font-size: 16px;
        }}
        .kpi-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .kpi-table tr:hover {{
            background-color: #f1f1f1;
        }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <style>
        /* 2. Texto NEGRO para las opciones en el cuadro principal */
        .stMultiSelect [data-baseweb="tag"] span,
        .stSelectbox [data-baseweb="select"] > div > div,
        .stMultiSelect [data-baseweb="select"] input,
        .stSelectbox [data-baseweb="select"] input {{
            color: black !important;
        }}

        /* 3. Fondo ROJO para el dropdown (lista desplegable) */
        .stMultiSelect [data-baseweb="popover"],
        .stSelectbox [data-baseweb="popover"] {{
            background-color: #e74c3c !important;  /* Rojo */
            border: 2px solid #c0392b !important;  /* Borde rojo oscuro */
        }}

        /* 4. Texto BLANCO en las opciones del dropdown */
        .stMultiSelect [data-baseweb="popover"] li,
        .stSelectbox [data-baseweb="popover"] li,
        .stMultiSelect [data-baseweb="popover"] div,
        .stSelectbox [data-baseweb="popover"] div {{
            color: white !important;
        }}

        /* 5. Efecto hover en las opciones (rojo más oscuro) */
        .stMultiSelect [data-baseweb="popover"] li:hover,
        .stSelectbox [data-baseweb="popover"] li:hover {{
            background-color: #c0392b !important;
        }}

        /* 6. Scrollbar del dropdown (opcional) */
        .stMultiSelect [data-baseweb="popover"]::-webkit-scrollbar-thumb,
        .stSelectbox [data-baseweb="popover"]::-webkit-scrollbar-thumb {{
            background-color: #c0392b !important;
        }}
    </style>
    """, unsafe_allow_html=True)
    

    main()
