"""Frontend regression tests using original customer records and mocked HTTP."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
import io
import json

import httpx
import numpy as np
import pandas as pd
import pytest

from frontend import config, batch_prediction as bp, risk_analytics as ra
from frontend import retention_simulator as rs
from frontend.api_client import TelecomAPIClient, APIClientError, APIConnectionError, APITimeoutError
from src.business_config import BASELINE_CONFIG

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def raw():
    return bp.load_uploaded_csv((ROOT / 'data/raw/telco_churn.csv').read_bytes())

@pytest.fixture
def customers(raw):
    return bp.validate_batch_dataframe(raw.head(12))[0]

@pytest.fixture
def info():
    return dict(model_name='xgboost', calibration_method='sigmoid', model_version='2', run_id='mock-run', max_batch_size=5)


def response_for(customers, threshold):
    predictions = []
    for customer in customers:
        # Deliberately controlled mock scores, not trained model estimates.
        p = (customer['tenure'] % 10) / 10
        label = int(p >= threshold)
        predictions.append(dict(customerID=customer['customerID'], churn_probability=p,
            raw_churn_probability=p, prediction=label, prediction_label='Churn' if label else 'No Churn',
            classification_threshold=threshold, model_name='xgboost', calibration_method='sigmoid'))
    n = sum(p['prediction'] for p in predictions)
    return dict(predictions=predictions, total_customers=len(customers), predicted_churn=n, predicted_no_churn=len(customers)-n)

@pytest.fixture
def client(info):
    c = Mock()
    c.get_model_info.return_value = info.copy()
    c.predict_batch.side_effect = lambda customers, classification_threshold: response_for(customers, classification_threshold)
    return c

@pytest.fixture
def results(client, customers, info):
    return bp.run_batch_prediction(client, customers, .5, info)

@pytest.fixture
def analytics(results):
    return ra.prepare_analytics_data(results)

@pytest.fixture
def http_client():
    return TelecomAPIClient(config.FrontendSettings('http://127.0.0.1:8000', 60., 5.))


def test_config_defaults(monkeypatch):
    for name in ('TELECOM_API_URL','TELECOM_API_TIMEOUT','TELECOM_HEALTH_TIMEOUT'):
        monkeypatch.delenv(name, raising=False)
    s = config.get_frontend_settings()
    assert s.api_url == 'http://127.0.0.1:8000'
    assert (s.request_timeout, s.health_timeout) == (60.,5.)


def test_config_override(monkeypatch):
    monkeypatch.setenv('TELECOM_API_URL',' https://example.test/ ')
    monkeypatch.setenv('TELECOM_API_TIMEOUT','20')
    monkeypatch.setenv('TELECOM_HEALTH_TIMEOUT','2')
    s = config.get_frontend_settings()
    assert (s.api_url,s.request_timeout,s.health_timeout) == ('https://example.test',20.,2.)

@pytest.mark.parametrize('name',['TELECOM_API_TIMEOUT','TELECOM_HEALTH_TIMEOUT'])
@pytest.mark.parametrize('value',['0','-1','nan','inf','invalid'])
def test_invalid_timeout(monkeypatch,name,value):
    monkeypatch.setenv(name,value)
    with pytest.raises(ValueError): config.get_frontend_settings()


def test_invalid_url(monkeypatch):
    monkeypatch.setenv('TELECOM_API_URL','ftp://example.test')
    with pytest.raises(ValueError): config.get_frontend_settings()

@pytest.mark.parametrize('method,endpoint,timeout,status',[('get_health','/health',5,200),('get_readiness','/ready',5,503),('get_model_info','/model/info',60,200)])
def test_get_endpoints(monkeypatch,http_client,method,endpoint,timeout,status):
    request = Mock(return_value=httpx.Response(status,json={'status':'test'}))
    monkeypatch.setattr(httpx,'request',request)
    assert getattr(http_client,method)() == {'status':'test'}
    assert request.call_args.kwargs['url'].endswith(endpoint)
    assert request.call_args.kwargs['timeout'] == timeout
    assert request.call_args.kwargs['method'] == 'GET'

@pytest.mark.parametrize('method,endpoint,key',[('predict','/predict','customer'),('predict_batch','/predict/batch','customers'),('explain','/explain','customer'),('explain_global','/explain/global','customers')])
def test_post_endpoints(monkeypatch,http_client,customers,method,endpoint,key):
    request = Mock(return_value=httpx.Response(200,json={'ok':True}))
    monkeypatch.setattr(httpx,'request',request)
    data = customers if key == 'customers' else customers[0]
    kwargs = {} if method == 'explain_global' else {'classification_threshold':0.0}
    assert getattr(http_client,method)(data,**kwargs) == {'ok':True}
    args = request.call_args.kwargs
    assert args['method'] == 'POST' and args['url'].endswith(endpoint)
    assert args['json'][key] == data
    if kwargs: assert args['json']['classification_threshold'] == 0.0
    if method == 'explain': assert args['json']['top_n'] == 10


def test_omitted_threshold(monkeypatch,http_client,customers):
    request = Mock(return_value=httpx.Response(200,json={}))
    monkeypatch.setattr(httpx,'request',request)
    http_client.predict(customers[0])
    assert 'classification_threshold' not in request.call_args.kwargs['json']

@pytest.mark.parametrize('error,expected',[(httpx.ConnectError('offline'),APIConnectionError),(httpx.ReadTimeout('slow'),APITimeoutError)])
def test_transport_errors(monkeypatch,http_client,error,expected):
    monkeypatch.setattr(httpx,'request',Mock(side_effect=error))
    with pytest.raises(expected): http_client.get_health()


def test_api_error_metadata(monkeypatch,http_client):
    monkeypatch.setattr(httpx,'request',Mock(return_value=httpx.Response(422,json={'message':'bad input','error':'validation'},headers={'X-Request-ID':'abc'})))
    with pytest.raises(APIClientError) as exc: http_client.get_health()
    assert (exc.value.status_code,exc.value.error_code,exc.value.request_id)==(422,'validation','abc')

@pytest.mark.parametrize('content',[b'not json',b'[]'])
def test_invalid_response(monkeypatch,http_client,content):
    monkeypatch.setattr(httpx,'request',Mock(return_value=httpx.Response(200,content=content)))
    with pytest.raises(APIClientError): http_client.get_health()


def test_original_dataset(raw):
    customers,ignored = bp.validate_batch_dataframe(raw)
    assert len(customers)==7043
    assert 'Churn' in ignored
    assert all('Churn' not in c for c in customers)
    assert len({c['customerID'] for c in customers})==7043

@pytest.mark.parametrize('content',[b'',b'\xff\xfe',b'\n'])
def test_bad_csv(content):
    with pytest.raises(ValueError): bp.load_uploaded_csv(content)

@pytest.mark.parametrize('case',['empty','missing','duplicate_id','duplicate_column','invalid_gender','negative_charge','blank_id','too_many'])
def test_invalid_batch(raw,monkeypatch,case):
    d=raw.head(3).copy()
    if case=='empty': d=d.iloc[:0]
    elif case=='missing': d=d.drop(columns='tenure')
    elif case=='duplicate_id': d.loc[d.index[1],'customerID']=d.iloc[0]['customerID']
    elif case=='duplicate_column': d=pd.concat([d,d[['tenure']]],axis=1)
    elif case=='invalid_gender': d.loc[d.index[0],'gender']='invalid'
    elif case=='negative_charge': d.loc[d.index[0],'MonthlyCharges']=-1
    elif case=='blank_id': d.loc[d.index[0],'customerID']=' '
    else: monkeypatch.setattr(bp,'MAX_UPLOAD_ROWS',2)
    with pytest.raises(ValueError): bp.validate_batch_dataframe(d)

@pytest.mark.parametrize('p,band',[(0,'Low'),(.249,'Low'),(.25,'Moderate'),(.499,'Moderate'),(.5,'High'),(.749,'High'),(.75,'Very High'),(1,'Very High')])
def test_risk_boundaries(p,band): assert bp.get_risk_band(p)==band

@pytest.mark.parametrize('p',[-.1,1.1,np.nan,np.inf,-np.inf])
def test_invalid_risk(p):
    with pytest.raises(ValueError): bp.get_risk_band(p)

@pytest.mark.parametrize('field,value',[('customerID','wrong'),('model_name','wrong'),('calibration_method','wrong'),('churn_probability',np.nan),('raw_churn_probability',2),('classification_threshold',.8),('prediction',2),('prediction_label','wrong')])
def test_response_integrity(customers,info,field,value):
    r=response_for(customers,.5)
    r['predictions'][0][field]=value
    with pytest.raises(APIClientError): bp.validate_batch_response(r,customers,info,.5)

@pytest.mark.parametrize('field',['total_customers','predicted_churn','predicted_no_churn'])
def test_response_counts(customers,info,field):
    r=response_for(customers,.5); r[field]+=1
    with pytest.raises(APIClientError): bp.validate_batch_response(r,customers,info,.5)


def test_batch_chunking(client,customers,info):
    progress=Mock()
    d=bp.run_batch_prediction(client,customers,.5,info,progress)
    assert [len(c.kwargs['customers']) for c in client.predict_batch.call_args_list]==[5,5,2]
    assert [c.args for c in progress.call_args_list]==[(5,12),(10,12),(12,12)]
    assert d.customerID.tolist()==[c['customerID'] for c in customers]
    assert d.model_version.eq('2').all()
    assert d.model_run_id.eq(info['run_id']).all()

@pytest.mark.parametrize('when',['before','during'])
def test_model_change(client,customers,info,when):
    changed=dict(info,model_version='3')
    client.get_model_info.side_effect=[changed] if when=='before' else [info,changed]
    with pytest.raises(APIClientError): bp.run_batch_prediction(client,customers,.5,info)

@pytest.mark.parametrize('threshold',[-1,2,np.nan,True])
def test_invalid_threshold(client,customers,info,threshold):
    with pytest.raises(ValueError): bp.run_batch_prediction(client,customers,threshold,info)


def test_batch_summary_export(results):
    s=bp.summarize_batch(results)
    assert s['total_customers']==len(results)
    assert s['predicted_churn']+s['predicted_no_churn']==len(results)
    assert s['expected_churn_count']==pytest.approx(results.churn_probability.sum())
    exported=pd.read_csv(io.BytesIO(bp.export_predictions(results)))
    assert exported.customerID.tolist()==results.customerID.tolist()


def test_analytics_math(results,analytics):
    assert 'tenure_group' not in results
    expected=results.churn_probability*results.MonthlyCharges
    np.testing.assert_allclose(analytics.expected_monthly_revenue_loss,expected)
    s=ra.calculate_portfolio_summary(analytics)
    assert s['expected_monthly_revenue_loss']==pytest.approx(expected.sum())
    assert s['predicted_churn']==results.prediction.sum()
    assert s['total_monthly_revenue']==pytest.approx(results.MonthlyCharges.sum())

@pytest.mark.parametrize('field,value',[('churn_probability',-1),('prediction',2),('risk_band','bad'),('MonthlyCharges',-1),('tenure',-1),('churn_probability',np.inf)])
def test_analytics_invalid(results,field,value):
    results.loc[results.index[0],field]=value
    with pytest.raises(ValueError): ra.prepare_analytics_data(results)


def test_risk_distribution(analytics):
    d=ra.calculate_risk_distribution(analytics)
    assert d.risk_band.tolist()==ra.RISK_ORDER
    assert d.customers.sum()==len(analytics)
    assert d.customer_share.sum()==pytest.approx(1)
    assert d.expected_revenue_loss.sum()==pytest.approx(analytics.expected_monthly_revenue_loss.sum())

@pytest.mark.parametrize('segment',list(ra.SEGMENT_COLUMNS))
def test_segment_totals(analytics,segment):
    d=ra.calculate_segment_summary(analytics,segment)
    assert d.customers.sum()==len(analytics)
    assert d.customer_share.sum()==pytest.approx(1)
    assert d.average_probability.is_monotonic_decreasing


def test_priority(analytics):
    d=ra.get_priority_customers(analytics,3)
    assert len(d)==3
    assert d.expected_monthly_revenue_loss.is_monotonic_decreasing
    assert d.iloc[0].expected_monthly_revenue_loss==analytics.expected_monthly_revenue_loss.max()

@pytest.mark.parametrize('limit',[0,-1,True,1.5])
def test_invalid_priority(analytics,limit):
    with pytest.raises(ValueError): ra.get_priority_customers(analytics,limit)


def test_shap_report():
    d,m=ra.load_global_shap_report(ROOT/'reports/shap')
    assert len(d)==m['feature_count']
    assert d['rank'].is_monotonic_increasing
    assert ra.shap_report_matches_model(m,m.copy())
    assert not ra.shap_report_matches_model(m,dict(m,model_version='different'))
    assert not ra.shap_report_matches_model(m,None)


def test_shap_tampering(tmp_path):
    source=ROOT/'reports/shap'
    (tmp_path/'shap_metadata.json').write_bytes((source/'shap_metadata.json').read_bytes())
    (tmp_path/'global_shap_importance.csv').write_bytes((source/'global_shap_importance.csv').read_bytes()+b'\n')
    with pytest.raises(ValueError,match='fingerprint'): ra.load_global_shap_report(tmp_path)


def test_economics(results):
    c=BASELINE_CONFIG
    d=rs.calculate_customer_economics(results,c)
    expected=results.churn_probability*results.MonthlyCharges*c.retention_success_rate*c.retention_months*c.contribution_margin-c.intervention_cost
    np.testing.assert_allclose(d.expected_net_benefit,expected)
    assert d.profitable_intervention.tolist()==(expected>0).tolist()

@pytest.mark.parametrize('budget',[0,25,50,10000])
def test_campaign_budget(results,budget):
    c=replace(BASELINE_CONFIG,campaign_budget=budget)
    summary,tables,economics=rs.compare_campaign_strategies(results,.5,c)
    s=summary.set_index('strategy').loc['budget_constrained']
    t=tables['budget_constrained']
    assert s.campaign_cost<=budget+1e-8
    assert len(t)<=int(budget/c.intervention_cost)
    assert (t.expected_net_benefit>0).all()
    assert s.expected_net_profit==pytest.approx(t.expected_net_benefit.sum())
    assert t.campaign_rank.tolist()==list(range(1,len(t)+1))
    expected=economics[economics.expected_net_benefit>0].sort_values(['expected_net_benefit','customerID'],ascending=[False,True]).head(int(budget/c.intervention_cost))
    assert t.customerID.tolist()==expected.customerID.tolist()


def test_threshold_campaign(results):
    s,t,_=rs.compare_campaign_strategies(results,.5)
    assert set(t['threshold'].customerID)==set(results.loc[results.churn_probability>=.5,'customerID'])
    assert len(s)==3


def test_zero_success(results):
    c=replace(BASELINE_CONFIG,retention_success_rate=0)
    s,t,_=rs.compare_campaign_strategies(results,.5,c)
    assert t['profitable'].empty and t['budget_constrained'].empty
    assert s.set_index('strategy').loc['budget_constrained','expected_net_profit']==0

@pytest.mark.parametrize('field,value',[('customerID',''),('customerID',None),('churn_probability',2),('MonthlyCharges',-1),('churn_probability',np.nan)])
def test_invalid_portfolio(results,field,value):
    results.loc[results.index[0],field]=value
    with pytest.raises(ValueError): rs.validate_portfolio(results)


def test_sensitivity(results):
    d=rs.run_campaign_sensitivity(results)
    assert len(d)==30
    assert not d.duplicated(['retention_success_rate','incentive_cost']).any()
    assert d.within_budget.all()
    assert (d.expected_net_profit>=-1e-8).all()
    for _,group in d.groupby('incentive_cost'):
        assert group.sort_values('retention_success_rate').expected_net_profit.is_monotonic_increasing
