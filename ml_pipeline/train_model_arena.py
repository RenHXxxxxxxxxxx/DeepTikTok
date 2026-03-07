import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from pathlib import Path

# *========== 全局配置 ==========*
GLOBAL_CONFIG = {
    # *特征列定义*
    "base_features": [
        "visual_brightness", 
        "visual_saturation", 
        "audio_bpm", 
        "cut_frequency",
        "follower_count",
        "comment_count",
        "collect_count",
        "share_count",
        "download_count"
    ],
    "categorical_feature": "theme_label",
    "target_feature": "digg_count",
    
    # *数据采样配置 (Stratified Replay Buffer)*
    "max_training_samples": 10000,
    "challenger_ratio": 0.7,
    "challenger_max_samples": 7000,
    "replay_buffer_max_samples": 3000,
    
    # *验证与部署关卡 (Testing & Deployment Gate)*
    "holdout_ratio": 0.15,
    "win_margin": 0.02,
    
    # *输出路径*
    "artifacts_dir": r"d:\renhangxi_tiktok_bysj\ml_pipeline\artifacts",
    "best_model_path": "best_model_2026.pkl",
    "manifest_path": "version_manifest.json",
    
    # *交叉验证参数*
    "n_splits": 5,
    "random_state": 42
}

def setup_django_env():
    try:
        # *1. 关键：将最外层工作区加入路径，这样才能识别 'renhangxi_tiktok_bysj' 这个包*
        workspace_root = r"D:\renhangxi_tiktok_bysj"
        project_root = os.path.join(workspace_root, "renhangxi_tiktok_bysj")
        
        if workspace_root not in sys.path:
            sys.path.insert(0, workspace_root)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # *2. 这里的路径必须与 INSTALLED_APPS 中的前缀一致*
        os.environ["DJANGO_SETTINGS_MODULE"] = "renhangxi_tiktok_bysj.settings"
        
        import django
        django.setup()
        print(f"✅ Django 环境初始化成功！已对齐命名空间。")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        raise

def load_data():
    try:
        from django.apps import apps
        from django.core.cache import cache
        
        Video = apps.get_model('douyin_hangxi', 'Video')
        
        # *获取全局 ACTIVE_TASK 以确定当前主题*
        active_task = cache.get('ACTIVE_TASK', {})
        active_theme = active_task.get('current_theme', None)
        
        fields = GLOBAL_CONFIG['base_features'] + [GLOBAL_CONFIG['categorical_feature'], GLOBAL_CONFIG['target_feature']]
        
        total_db_records = Video.objects.filter(analysis_status=2).count()
        if total_db_records == 0:
            raise ValueError("*数据库中 analysis_status=2 的数据量为 0，请先进行爬取分析！*")
            
        # *动态调整 70/30 比例*
        if total_db_records < GLOBAL_CONFIG['max_training_samples']:
            target_challenger = int(total_db_records * GLOBAL_CONFIG['challenger_ratio'])
            target_replay = total_db_records - target_challenger
        else:
            target_challenger = GLOBAL_CONFIG['challenger_max_samples']
            target_replay = GLOBAL_CONFIG['replay_buffer_max_samples']
            
        if active_theme:
            # *Challenger Data (New Trends): 按时间降序获取最新数据*
            challenger_qs = Video.objects.filter(
                theme_label=active_theme, 
                analysis_status=2
            ).order_by('-create_time').values(*fields)[:target_challenger]
            df_challenger = pd.DataFrame(challenger_qs)
            
            # *Replay Buffer (Global Historical Baseline): 随机采样其他主题*
            replay_qs = Video.objects.filter(
                analysis_status=2
            ).exclude(
                theme_label=active_theme
            ).order_by('?').values(*fields)[:target_replay]
            df_replay = pd.DataFrame(replay_qs)
            
            # *合并数据*
            df_final = pd.concat([df_challenger, df_replay], ignore_index=True)
        else:
            # *后备方案: 若无明确主题则全局采样*
            qs = Video.objects.filter(analysis_status=2).order_by('?').values(*fields)[:GLOBAL_CONFIG['max_training_samples']]
            df_final = pd.DataFrame(qs)
            
        if df_final.empty:
            raise ValueError("*特征合并后数据为空！*")
            
        # *混合打乱 (Shuffle)*
        df_final = df_final.sample(frac=1, random_state=GLOBAL_CONFIG['random_state']).reset_index(drop=True)
        
        print(f"\n[{'*' * 20} Stratified Replay Buffer {'*' * 20}]")
        print(f"*Active Theme*: {active_theme}")
        print(f"*Total DB Records*: {total_db_records}")
        if active_theme:
            print(f"*Challenger Data*: {len(df_challenger)} samples")
            print(f"*Replay Buffer*: {len(df_replay)} samples")
        print(f"*Final Training Payload*: {len(df_final)} samples\n")
        
        return df_final
    except Exception as e:
        print(f"*数据加载失败*: {e}")
        # *打印调试信息，确认模型名*
        from django.apps import apps
        print(f"   - 检查 INSTALLED_APPS: {[app.label for app in apps.get_app_configs()]}")
        raise

def feature_engineering(df):
    # *严格的特征工程管道 (避免拓扑遗忘)*
    try:
        # *分离目标变量*
        y = df[GLOBAL_CONFIG['target_feature']].copy()
        
        # *基础物理特征*
        base_df = df[GLOBAL_CONFIG['base_features']].copy()
        
        # *缺失值处理 (中位数策略)*
        imputer = SimpleImputer(strategy='median')
        
        # *保持 DataFrame 格式不丢失列名*
        base_imputed = pd.DataFrame(imputer.fit_transform(base_df), columns=base_df.columns, index=base_df.index)
        
        # *分类特征独热编码*
        cat_df = df[[GLOBAL_CONFIG['categorical_feature']]].copy()
        cat_encoded = pd.get_dummies(cat_df, columns=[GLOBAL_CONFIG['categorical_feature']], dummy_na=False)
        cat_encoded = cat_encoded.astype(float)
        
        # *特征组装*
        X = pd.concat([base_imputed, cat_encoded], axis=1)
        
        # *显式保留最新的特征列名称，防止拓扑遗忘*
        feature_names_in_ = X.columns.tolist()
        
        # *标准化处理*
        scaler = StandardScaler()
        
        # *必须依旧输出 DataFrame，绝不提前转为 NumPy 数组传给 fit()*
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names_in_, index=X.index)
        
        return X_scaled, y, feature_names_in_, scaler
    except Exception as e:
        print(f"*特征工程处理失败*: {e}")
        raise

def train_and_evaluate(X, y):
    # *模型竞技场及交叉验证*
    try:
        # *全局验证集切分 (Global Hold-out Split)*
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, 
            test_size=GLOBAL_CONFIG['holdout_ratio'], 
            random_state=GLOBAL_CONFIG['random_state']
        )

        models = {
            "XGBRegressor": XGBRegressor(
                random_state=GLOBAL_CONFIG['random_state'], 
                n_estimators=100, 
                learning_rate=0.1
            ),
            "RandomForestRegressor": RandomForestRegressor(
                random_state=GLOBAL_CONFIG['random_state'], 
                n_estimators=100
            )
        }
        
        kf = KFold(n_splits=GLOBAL_CONFIG['n_splits'], shuffle=True, random_state=GLOBAL_CONFIG['random_state'])
        scoring = {'r2': 'r2', 'neg_rmse': 'neg_root_mean_squared_error'}
        
        results = {}
        
        # *打印控制台摘要头*
        print(f"\n[{'*' * 20} 模型竞技场交叉验证评估 (5-Fold) {'*' * 20}]")
        print(f"*样本总数*: {len(X)}")
        print(f"*Training Set*: {len(X_train)} samples")
        print(f"*Hold-out Validation Set*: {len(X_holdout)} samples")
        print(f"*特征维度*: {X_train.shape[1]}")
        
        for name, model in models.items():
            # *仅在 Training Set 上做 CV*
            cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring, return_estimator=True)
            
            mean_rmse = -cv_results['test_neg_rmse'].mean()
            mean_r2 = cv_results['test_r2'].mean()
            
            # *在全量训练数据上重训作为待选资产 (不包含 Hold-out)*
            model.fit(X_train, y_train)
            
            results[name] = {
                "mean_rmse": mean_rmse,
                "mean_r2": mean_r2,
                "fitted_model": model
            }
            
            # *打印模型得分*
            print(f"\n[{name}]")
            print(f"  *CV RMSE*: {mean_rmse:.4f}")
            print(f"  *CV R^2* : {mean_r2:.4f}")
            
        return results, X_holdout, y_holdout
    except Exception as e:
        print(f"*模型训练失败*: {e}")
        raise

def load_champion(artifacts_dir):
    # *加载现任冠军模型*
    try:
        manifest_path = Path(artifacts_dir) / GLOBAL_CONFIG['manifest_path']
        if not manifest_path.exists():
            return None
            
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
            
        model_file = manifest.get("model_file")
        if not model_file:
            return None
            
        model_path = Path(artifacts_dir) / model_file
        if not model_path.exists():
            return None
            
        champion_model = joblib.load(model_path)
        return champion_model, manifest
    except Exception as e:
        print(f"*加载 Champion 失败，可能需要自动降级*: {e}")
        return None

def champion_vs_challenger(champion_model, challenger_model, X_holdout, y_holdout):
    # *冠军挑战者终极对决*
    try:
        print(f"\n[{'*' * 20} Champion VS Challenger 终极对决 {'*' * 20}]")
        
        # *Champion 预测*
        # *处理可能的特征空间不匹配情况 (拓扑防御)*
        if hasattr(champion_model, 'feature_names_in_'):
            champ_features = champion_model.feature_names_in_
            # *仅提取 Champion 认识的特征，缺失的补 0*
            X_champ = pd.DataFrame(index=X_holdout.index)
            for col in champ_features:
                if col in X_holdout.columns:
                    X_champ[col] = X_holdout[col]
                else:
                    X_champ[col] = 0
            y_pred_champ = champion_model.predict(X_champ)
        else:
            y_pred_champ = champion_model.predict(X_holdout)
            
        champ_rmse = np.sqrt(mean_squared_error(y_holdout, y_pred_champ))
        champ_r2 = r2_score(y_holdout, y_pred_champ)
        
        # *Challenger 预测*
        y_pred_challenger = challenger_model.predict(X_holdout)
        challenger_rmse = np.sqrt(mean_squared_error(y_holdout, y_pred_challenger))
        challenger_r2 = r2_score(y_holdout, y_pred_challenger)
        
        print(f"*Champion RMSE*  : {champ_rmse:.4f}  | *R²*: {champ_r2:.4f}")
        print(f"*Challenger RMSE*: {challenger_rmse:.4f}  | *R²*: {challenger_r2:.4f}")
        
        margin = GLOBAL_CONFIG['win_margin']
        margin_diff = champ_rmse - challenger_rmse
        
        print(f"*Required Margin*: {margin}")
        print(f"*Actual Margin*  : {margin_diff:.4f}")
        
        # *挑战者必须比冠军的 RMSE 小一定的 Margin 才算赢*
        wins = margin_diff >= margin
        
        metrics_dict = {
            "champ_rmse": float(champ_rmse),
            "champ_r2": float(champ_r2),
            "challenger_rmse": float(challenger_rmse),
            "challenger_r2": float(challenger_r2),
            "margin_diff": float(margin_diff)
        }
        
        return wins, metrics_dict
    except Exception as e:
        print(f"*对决过程异常，阻止部署*: {e}")
        return False, {}

def save_artifacts(results, feature_names, scaler, holdout_metrics=None):
    # *模型资产持久化 — 包含 Hold-out 指标*
    try:
        # *按 CV RMSE 选出最佳 Challenger*
        best_model_name = min(results, key=lambda k: results[k]['mean_rmse'])
        best_model = results[best_model_name]['fitted_model']
        best_rmse = results[best_model_name]['mean_rmse']
        best_r2 = results[best_model_name]['mean_r2']

        print(f"\n[{'*' * 20} 部署新模型: {best_model_name} (CV RMSE: {best_rmse:.4f}) {'*' * 20}]")

        art_dir = Path(GLOBAL_CONFIG['artifacts_dir'])
        art_dir.mkdir(parents=True, exist_ok=True)

        version_id = datetime.now().strftime('v%Y%m%d_%H%M%S')
        model_abbr = best_model_name.replace('Regressor', '').replace('RandomForest', 'RF')
        model_filename = f"model_{model_abbr}_{version_id}.pkl"
        scaler_filename = f"scaler_{model_abbr}_{version_id}.pkl"

        model_path = art_dir / model_filename
        scaler_path = art_dir / scaler_filename
        manifest_path = art_dir / GLOBAL_CONFIG['manifest_path']

        # *提取旧版本的版号并记录，以支持零宕机回滚*
        previous_version = None
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    old_manifest = json.load(f)
                    previous_version = old_manifest.get("active_version")
            except Exception as e:
                print(f"*读取现存清单失败*: {e}")

        # *提取超参数并进行 JSON 序列化安全转换*
        hyperparameters = {}
        if hasattr(best_model, 'get_params'):
            try:
                raw_params = best_model.get_params()
                for k, v in raw_params.items():
                    if isinstance(v, (int, float, str, bool, type(None))):
                        hyperparameters[k] = v
                    elif hasattr(v, 'item'):
                        hyperparameters[k] = v.item()
                    else:
                        hyperparameters[k] = str(v)
            except Exception as e:
                print(f"*超参数提取失败*: {e}")
                    
        # *提取特征重要性 (Top 15)*
        feature_importances_dict = {}
        if hasattr(best_model, 'feature_importances_'):
            try:
                importances = best_model.feature_importances_
                # *打包并按重要性降序排序*
                sorted_idx = np.argsort(importances)[::-1]
                for idx in sorted_idx[:15]:
                    feature_importances_dict[feature_names[idx]] = float(importances[idx])
            except Exception as e:
                print(f"*特征重要性提取失败*: {e}")

        manifest = {
            "active_version": version_id,
            "previous_version": previous_version,
            "model_type": best_model_name,
            "model_file": model_filename,
            "scaler_file": scaler_filename,
            "trained_at": datetime.now().isoformat(),
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "hyperparameters": hyperparameters,
            "feature_importances": feature_importances_dict,
            "metrics": {
                "cv_rmse": float(best_rmse),
                "cv_r2": float(best_r2),
                "holdout_metrics": holdout_metrics or {}
            }
        }

        # *1. 首先写入临时文件以避免部署时文件锁定或损坏*
        model_temp_path = art_dir / "model_temp.pkl"
        scaler_temp_path = art_dir / "scaler_temp.pkl"
        manifest_temp_path = art_dir / "manifest_temp.json"

        joblib.dump(best_model, model_temp_path)
        joblib.dump(scaler, scaler_temp_path)
        with open(manifest_temp_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=4)

        # *2. 使用系统级的原子操作瞬间覆盖，杜绝脏读*
        os.replace(model_temp_path, model_path)
        os.replace(scaler_temp_path, scaler_path)
        os.replace(manifest_temp_path, manifest_path)

        # *3. 自动回收过期的资产，仅保留最近三个版本*
        try:
            all_models = sorted(art_dir.glob("model_*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
            for old_model in all_models[3:]:
                try:
                    old_model.unlink()
                except Exception:
                    pass
                
            all_scalers = sorted(art_dir.glob("scaler_*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
            for old_scaler in all_scalers[3:]:
                try:
                    old_scaler.unlink()
                except Exception:
                    pass
        except Exception as e:
            print(f"*空间回收失败*: {e}")

        print(f"*最佳模型已原子性写入至*: {model_path}")
        print(f"*标准化器已原子性写入至*: {scaler_path}")
        print(f"*版本特征拓扑清单已被原子保护至*: {manifest_path}\n")

    except Exception as e:
        print(f"*资产保存失败*: {e}")
        raise

def main():
    # *执行主训练管道：包含竞技场对比*
    try:
        setup_django_env()
        df = load_data()
        X, y, feature_names, scaler = feature_engineering(df)
        
        # *阶段一：训练 Challenger*
        results, X_holdout, y_holdout = train_and_evaluate(X, y)
        
        best_name = min(results, key=lambda k: results[k]['mean_rmse'])
        challenger_model = results[best_name]['fitted_model']
        
        # *阶段二：加载 Champion*
        champion_data = load_champion(GLOBAL_CONFIG['artifacts_dir'])
        
        # *阶段三：部署决策*
        if champion_data is None:
            print("\n[The Arena] *首个模型！没有发现可用 Champion，自动提升 Challenger.*")
            
            # *由于是首个模型，其在 Holdout 集上的表现可以直接记录*
            y_pred = challenger_model.predict(X_holdout)
            holdout_metrics = {
                "note": "First run, auto-promoted.",
                "challenger_rmse": float(np.sqrt(mean_squared_error(y_holdout, y_pred))),
                "challenger_r2": float(r2_score(y_holdout, y_pred))
            }
            save_artifacts(results, feature_names, scaler, holdout_metrics)
        else:
            champion_model, manifest = champion_data
            print(f"\n[The Arena] *发现现任 Champion*: {manifest.get('model_type')} ({manifest.get('active_version')})")
            
            wins, metrics = champion_vs_challenger(champion_model, challenger_model, X_holdout, y_holdout)
            
            if wins:
                print("\n[The Arena] 🏆 *Challenger 挑战成功！覆盖旧模型...*")
                save_artifacts(results, feature_names, scaler, metrics)
            else:
                print("\n[The Arena] ⛔ *Challenger 挑战失败！没有达到部署要求(Win Margin).*")
                print("*本次部署被中止，系统保持现有 Champion 版本！*\n")
                sys.exit(0)
                
    except Exception as e:
        print(f"\n[FATAL] *训练管道执行崩溃*: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
