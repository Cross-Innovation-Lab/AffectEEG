"""Architecture parity, one-update smoke, splits and train-only adaptation checks."""
import os
import importlib.util
import numpy as np
import pytest
import torch
from affecteeg.tone import protocol as p, adaptation as a, adaptation_math as am
from affecteeg.tone.models import build_model
from affecteeg.tone.interfaces import assert_identity_parity, FeatureExtractor, contract_from_model
from affecteeg.tone.training import ToneDataset, _reference, fit_affect
from affecteeg.tone.io import save_checkpoint, load_checkpoint


def _device():
    device = torch.device(os.environ.get('AFFECTEEG_TEST_DEVICE', 'cpu'))
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    return device


@pytest.mark.parametrize('backbone', [x for x in p.BACKBONES if x != 'eegmamba'])
def test_native_backbone_interface_and_training_update(backbone):
    device = _device()
    torch.manual_seed(42)
    model = build_model(backbone).to(device)
    eeg = torch.randn(2, 499 if backbone == 'cat_net' else 500,
                      10 if backbone == 'cat_net' else 20, device=device)
    emg = torch.randn(2, eeg.shape[1], 10 if backbone == 'cat_net' else 0, device=device)
    assert_identity_parity(backbone, model, eeg, emg)
    state = FeatureExtractor(backbone, model).extract(eeg, emg)
    assert state['target'].shape[-1] == p.FEATURE_DIMS[backbone]
    model.train()
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad], lr=.001)
    logits, _ = model(eeg, emg)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0,1], device=device))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(v.grad is not None and bool(v.grad.abs().sum() > 0) for v in model.parameters())
    optimizer.step()
    assert all(bool(torch.isfinite(v).all()) for v in model.parameters())


def test_eegmamba_cuda_update():
    if not torch.cuda.is_available() or importlib.util.find_spec('mamba_ssm') is None:
        pytest.skip('Official EEGMamba requires CUDA and mamba-ssm')
    device = torch.device(os.environ.get('AFFECTEEG_TEST_DEVICE', 'cuda'))
    if device.type != 'cuda':
        pytest.skip('Set AFFECTEEG_TEST_DEVICE=cuda:<index> for the CUDA smoke check')
    torch.cuda.set_device(device)
    model = build_model('eegmamba').to(device)
    eeg = torch.randn(2, 500, 20, device=device)
    emg = torch.empty(2, 500, 0, device=device)
    assert_identity_parity('eegmamba', model, eeg, emg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    with torch.amp.autocast('cuda'):
        loss = torch.nn.functional.cross_entropy(model(eeg, emg)[0], torch.tensor([0, 1], device=device))
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert all(bool(torch.isfinite(v).all()) for v in model.parameters())


def test_seed_specific_participant_splits():
    subjects = np.repeat(np.arange(1,11),480)
    for seed, offset in p.OFFSETS.items():
        for fold in range(1,11):
            part = p.split(subjects, seed, fold)
            assert [len(part[k]) for k in ('train','validation','test')] == [3840,480,480]
            assert set(subjects[part['validation']]) == {(fold-1+offset)%10+1}
            assert set(subjects[part['test']]) == {fold}
            assert not set(part['train']) & set(part['validation'])
            assert not set(part['train']) & set(part['test'])


def test_exact_signal_and_difference_alignment():
    names = list(dict.fromkeys((*p.EEG20,*p.CAT_CHANNELS)))
    data = dict(eeg=np.arange(500*len(names),dtype=np.float32).reshape(1,500,-1),
        emg=np.arange(2500,dtype=np.float32).reshape(1,500,5),labels=[0],subjects=[1],
        metadata={'channels': names})
    cat = ToneDataset(data,[0],'cat_net')[0]
    selected = data['eeg'][0][:,[names.index(c) for c in p.CAT_CHANNELS]]
    np.testing.assert_array_equal(cat[0][:,:5],selected[1:])
    np.testing.assert_array_equal(cat[0][:,5:],np.diff(selected,axis=0))
    assert ToneDataset(data,[0],'eegnet')[0][0].shape == (500,20)


def test_adapter_zero_initialization_statistics_and_covariance():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(32,5,8)).astype(np.float32)
    x[:,:,0] = 2
    stats = am.feature_statistics(x)
    adapter = am.RelativeAdapter(stats,rank=4,scale=.1)
    tx = torch.tensor(x)
    assert torch.equal(adapter(tx),tx)
    assert not stats['active'][:,0].any()
    with torch.no_grad():
        adapter.up.weight.fill_(.2)
    assert torch.equal(adapter(tx)[:,:,0],tx[:,:,0])
    labels = np.tile(np.arange(4),8)
    teacher = rng.normal(size=(32,12)).astype(np.float32)+labels[:,None]*10
    residual, state = am.teacher_targets(teacher,labels)
    for tone in range(4):
        np.testing.assert_allclose(residual[labels==tone].mean(0),0,atol=1e-6)
    cov = am.covariance_penalty(torch.tensor(x.reshape(32,-1)),torch.tensor(residual))
    direct = (torch.tensor(x.reshape(32,-1)).T @ torch.tensor(residual)/32).square().mean()
    assert torch.allclose(cov,direct,atol=1e-6)


def test_selection_ties_and_full_search_grid():
    rows = [dict(candidate_id='a', selected_epoch=1,validation_metrics={'balanced_accuracy':.5,'macro_f1':.5}),
            dict(candidate_id='z', selected_epoch=0,validation_metrics={'balanced_accuracy':.5,'macro_f1':.5})]
    assert p.select(rows)['selected_epoch'] == 0
    for name in p.BACKBONES:
        configs = p.templates(name)
        assert len(configs) == 14
        assert all(c['feature_mode']=='native' for c in configs)
    assert len(p.structural_templates('cat_net')) == 2


def test_reference_and_candidate_frozen_tail_checkpoint(tmp_path):
    _device()
    model = build_model('eegnet').eval().requires_grad_(False)
    contract = contract_from_model('eegnet',model)
    rng = np.random.default_rng(42)
    train = {'target':rng.normal(size=(128,240)).astype(np.float32)}
    val = {'target':rng.normal(size=(16,240)).astype(np.float32)}
    y, vy = np.tile(np.arange(4),32), np.tile(np.arange(4),4)
    stats = am.feature_statistics(train['target'])
    ref = _reference(contract,train,val,y,vy,stats,42,'cpu',1)
    config = p.templates('eegnet')[0]
    nuisance,_ = a.nuisance_view(rng.normal(size=(128,16)),y,'original')
    result = a.fit(contract=contract,train=train,validation=val,train_labels=y,
        validation_labels=vy,nuisance=nuisance,statistics=stats,config=config,
        reference=ref,arm='arra',calibration={'lambda':10.},seed=42,epochs=1)
    assert result['optimizer_steps'] == 1
    assert all(not v.requires_grad for v in model.parameters())
    fitted = a.FitModel(contract,stats,config,ref,'arra')
    assert all(not v.requires_grad for v in fitted.tail.parameters())
    before = a.predict(result,contract,val)
    path = tmp_path/'selected.pt';save_checkpoint(path,result)
    after = a.predict(load_checkpoint(path),contract,val)
    np.testing.assert_array_equal(before,after)


@pytest.mark.parametrize('backbone', ['cat_net','eeg_conformer','reve','eeg_deformer'])
def test_structural_interface_zero_adapter_and_update(backbone):
    from affecteeg.tone.structures import StructuralModel, cached_transform
    device = _device()
    model = build_model(backbone).to(device).eval().requires_grad_(False)
    eeg = torch.randn(2,499 if backbone=='cat_net' else 500,10 if backbone=='cat_net' else 20,device=device)
    emg = torch.randn(2,eeg.shape[1],10 if backbone=='cat_net' else 0,device=device)
    contract = contract_from_model(backbone, model)
    mode = p.STRUCTURAL_MODES[backbone]
    with torch.no_grad():
        native = model(eeg,emg)[0].cpu().numpy()
        state = {k:v.cpu().numpy() for k,v in FeatureExtractor(backbone,model).extract(eeg,emg).items()}
        if mode == 'conformer256':
            tokens = model.backbone.encoder(model.backbone.embedding(eeg.permute(0,2,1).unsqueeze(1)))
            state = {'target':model.backbone.classifier[:3](tokens).cpu().numpy()}
            contract['structural_tail_state'] = am._cpu_state(model.backbone.classifier[3:7])
        else:
            state = cached_transform(state,contract,mode)
    config = p.structural_templates(backbone)[0]
    stats = am.feature_statistics(state['target'])
    ref = {'head_state':contract['head']['state_dict']}
    fitted = a.make_model(contract,stats,config,ref,'arra',42,device,StructuralModel)
    np.testing.assert_allclose(a.logits(fitted,a.tensors(state,device)),native,atol=2e-5,rtol=2e-5)
    assert all(not v.requires_grad for v in fitted.tail.parameters())
    before = {k:v.detach().clone() for k,v in fitted.tail.state_dict().items()}
    optimizer = torch.optim.AdamW([v for v in fitted.parameters() if v.requires_grad],lr=3e-4)
    logits,clean = fitted(a.tensors(state,device),slice(None))
    loss = torch.nn.functional.cross_entropy(logits,torch.tensor([0,1],device=device))
    loss.backward();optimizer.step()
    assert torch.isfinite(loss)
    for key,value in fitted.tail.state_dict().items():
        assert torch.equal(before[key],value)


def test_calibrated_conditional_candidate():
    _device()
    model = build_model('eegnet').eval()
    contract = contract_from_model('eegnet',model)
    rng = np.random.default_rng(123)
    train = {'target':rng.normal(size=(128,240)).astype(np.float32)}
    val = {'target':rng.normal(size=(16,240)).astype(np.float32)}
    y,vy = np.tile(np.arange(4),32),np.tile(np.arange(4),4)
    stats = am.feature_statistics(train['target'])
    ref = _reference(contract,train,val,y,vy,stats,42,'cpu',1)
    config = p.templates('eegnet')[7]
    nuisance,transform = a.nuisance_view(rng.normal(size=(128,48)),y,config['teacher_view'])
    assert nuisance.shape == (128,32)
    assert transform['whitening'] is False
    calibration = a.calibrate(contract,train,y,nuisance,stats,config,ref,42,'cpu')
    assert .1 <= calibration['lambda'] <= 300
    assert len(calibration['batches']) == 8
    result = a.fit(contract=contract,train=train,validation=val,train_labels=y,
        validation_labels=vy,nuisance=nuisance,statistics=stats,config=config,
        reference=ref,arm='arra',calibration=calibration,seed=42,epochs=1)
    assert result['optimizer_steps'] == 1
    assert result['precision'] == 'float32'


def test_checkpoint_coordinate_and_partition_rejection():
    from affecteeg.tone.training import validate_coordinate
    import copy
    part = dict(train=np.arange(128),validation=np.arange(128,144),test=np.arange(144,160))
    saved = dict(backbone='eegnet',seed=42,fold=1,smoke=True,partitions=part)
    validate_coordinate(saved,'eegnet',42,1,True,part)
    for key,value in [('backbone','atcnet'),('seed',123),('fold',2),('smoke',False)]:
        wrong = dict(saved,**{key:value})
        with pytest.raises(ValueError,match='coordinate mismatch'):
            validate_coordinate(wrong,'eegnet',42,1,True,part)
    wrong = copy.deepcopy(saved)
    wrong['partitions']['test'][0] = 0
    with pytest.raises(ValueError,match='partitions differ'):
        validate_coordinate(wrong,'eegnet',42,1,True,part)
