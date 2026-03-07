/**
 * *全局状态轮询引擎*
 * *负责监控系统心跳并更新导航栏状态胶囊*
 */

// *全局 API 配置 (若外部已定义则复用)*
const GLOBAL_API_CONFIG = window.GLOBAL_API_CONFIG || {
    predictApi: '/predict/api/',
    globalStatus: '/api/global-status/',
    analysisStatus: '/api/get_analysis_status/'
};

(function () {
    'use strict';

    // *全局配置*
    const GLOBAL_CONFIG = {
        pollInterval: 2000,           // *轮询间隔 (毫秒)*
        apiEndpoint: GLOBAL_API_CONFIG.globalStatus,
        maxFailuresBeforeUnlock: 3,   // *连续失败次数上限，超过后自动解锁 UI*
        elementIds: {
            capsule: 'nav-status-capsule',
            statusText: 'status-text',
            progressBar: 'status-progress-bar',
            finishToast: 'finishToast',
            toastMessage: 'toast-message'
        },
        // *UI 锁定相关元素选择器*
        lockdownSelectors: {
            playBtn: '#global-play-btn',
            commandMenu: '#commandMenuBtn',
            deleteButtons: '.btn-delete-item, .btn-edit-item, [data-action="delete"], [data-action="edit"]',
            topNavbar: '.top-navbar'
        },
        // *锁定状态 CSS 类*
        lockdownClasses: {
            disabled: 'ui-lockdown-disabled',
            taskGlow: 'task-in-progress-glow'
        }
    };

    // *缓存上一次的状态数据用于对比*
    let lastStatusData = null;

    // *跟踪上一次爬虫运行状态，用于检测状态转换触发 Toast*
    let previousRunningState = null;

    // *连续 API 请求失败计数 (用于优雅恢复)*
    let consecutiveFailures = 0;

    // *当前锁定状态缓存 (防止重复 DOM 操作)*
    let currentLockdownState = null;

    /**
     * *深度比较两个对象是否相等*
     * @param {Object} obj1 
     * @param {Object} obj2 
     * @returns {boolean}
     */
    function deepEqual(obj1, obj2) {
        if (obj1 === obj2) return true;
        if (obj1 === null || obj2 === null) return false;
        if (typeof obj1 !== 'object' || typeof obj2 !== 'object') return false;

        const keys1 = Object.keys(obj1);
        const keys2 = Object.keys(obj2);

        if (keys1.length !== keys2.length) return false;

        for (const key of keys1) {
            if (!keys2.includes(key)) return false;
            if (!deepEqual(obj1[key], obj2[key])) return false;
        }

        return true;
    }

    /**
     * *计算进度百分比*
     * @param {number} current - *当前进度值*
     * @param {number} total - *总数值*
     * @returns {number} - *百分比 (0-100)*
     */
    function calculatePercentage(current, total) {
        if (!total || total <= 0) return 0;
        const percentage = (current / total) * 100;
        return Math.min(Math.max(percentage, 0), 100); // *限制在 0-100 范围*
    }

    /**
     * *更新 DOM 元素显示爬虫运行状态*
     * @param {Object} data - *API 返回的状态数据*
     */
    function updateDOMWithStatus(data) {
        // *Capsule elements*
        const capsule = document.getElementById(GLOBAL_CONFIG.elementIds.capsule);
        const statusText = document.getElementById(GLOBAL_CONFIG.elementIds.statusText);
        const progressBar = document.getElementById(GLOBAL_CONFIG.elementIds.progressBar);

        // *Radar Stepper elements*
        const stepVideo = document.getElementById('step-video');
        const stepComment = document.getElementById('step-comment');
        const stepAi = document.getElementById('step-ai');
        const radarProgressBar = document.getElementById('radar-progress-bar');
        const radarMsg = document.getElementById('radar-msg');

        let percent = 0;

        if (data.status === 'finished') {
            if (stepVideo) stepVideo.classList.add('active');
            if (stepComment) stepComment.classList.add('active');
            if (stepAi) stepAi.classList.add('active');
            percent = 100;
            if (window.radarInterval) {
                clearInterval(window.radarInterval);
            }
            if (window.modalPollingTimeout) {
                clearTimeout(window.modalPollingTimeout);
            }
        } else {
            if (data.global_phase === 'video_crawling') {
                if (stepVideo) stepVideo.classList.add('active');
                if (stepComment) stepComment.classList.remove('active');
                if (stepAi) stepAi.classList.remove('active');
                const c = (data.video && data.video.c) ? data.video.c : 0;
                const t = (data.video && data.video.t) ? data.video.t : 0;
                percent = t > 0 ? (c / t) * 100 : 0;
            } else if (data.global_phase === 'comment_crawling') {
                if (stepVideo) stepVideo.classList.remove('active');
                if (stepComment) stepComment.classList.add('active');
                if (stepAi) stepAi.classList.remove('active');
                const c = (data.comment && data.comment.c) ? data.comment.c : 0;
                const t = (data.comment && data.comment.t) ? data.comment.t : 0;
                percent = t > 0 ? (c / t) * 100 : 0;
            } else if (data.global_phase === 'ai_processing') {
                if (stepVideo) stepVideo.classList.remove('active');
                if (stepComment) stepComment.classList.remove('active');
                if (stepAi) stepAi.classList.add('active');
                const c = (data.ai && data.ai.c) ? data.ai.c : 0;
                const t = (data.ai && data.ai.t) ? data.ai.t : 0;
                percent = t > 0 ? (c / t) * 100 : 0;
            }
        }

        if (radarProgressBar) {
            radarProgressBar.style.width = percent + '%';
            radarProgressBar.innerText = Math.round(percent) + '%';
        }
        if (radarMsg && data.msg) {
            radarMsg.innerText = data.msg;
        }

        // *Fallback Capsule tracking*
        if (capsule && statusText && progressBar) {
            if (data.status !== 'finished' && (data.spider_running || data.global_phase)) {
                capsule.classList.remove('d-none');
                capsule.classList.add('d-flex');
                statusText.textContent = data.msg || '运行中...';
                progressBar.style.width = percent + '%';
                progressBar.setAttribute('aria-valuenow', Math.round(percent));

                progressBar.classList.remove('bg-success', 'bg-warning', 'bg-info');
                if (percent >= 80) progressBar.classList.add('bg-success');
                else if (percent >= 40) progressBar.classList.add('bg-info');
                else progressBar.classList.add('bg-warning');
            } else {
                capsule.classList.add('d-none');
                capsule.classList.remove('d-flex');
            }
        }
    }

    /**
     * *触发任务完成 Toast 通知*
     * @param {string} message - *Toast 消息内容*
     */
    function showFinishToast(message) {
        const toastEl = document.getElementById(GLOBAL_CONFIG.elementIds.finishToast);
        const toastMsg = document.getElementById(GLOBAL_CONFIG.elementIds.toastMessage);

        if (!toastEl) {
            console.warn('[GlobalTracker] *Toast 元素未找到*');
            return;
        }

        // *更新消息内容*
        if (toastMsg && message) {
            toastMsg.textContent = message;
        }

        // *初始化并显示 Toast*
        console.log('[Notification] Task finish detected. Triggering Toast.');
        const toast = new bootstrap.Toast(toastEl);
        toast.show();
    }

    /**
     * *响应式 UI 锁定：基于 spider_running 状态控制交互元素*
     * @param {boolean} isLocked - *true = 锁定 UI, false = 解锁 UI*
     */
    function applyUILockdown(isLocked) {
        // *防止重复 DOM 操作：状态未变化时直接返回*
        if (currentLockdownState === isLocked) {
            return;
        }
        currentLockdownState = isLocked;

        const selectors = GLOBAL_CONFIG.lockdownSelectors;
        const classes = GLOBAL_CONFIG.lockdownClasses;

        // *Action A: 处理全局播放按钮*
        const playBtn = document.querySelector(selectors.playBtn);
        if (playBtn) {
            const iconEl = playBtn.querySelector('i');
            if (isLocked) {
                playBtn.disabled = true;
                playBtn.style.pointerEvents = 'none';
                playBtn.classList.add(classes.disabled);
                if (iconEl) {
                    iconEl.className = 'fas fa-spinner fa-spin';
                }
            } else {
                playBtn.disabled = false;
                playBtn.style.pointerEvents = '';
                playBtn.classList.remove(classes.disabled);
                if (iconEl) {
                    iconEl.className = 'fas fa-play';
                }
            }
        }

        // *Action B: 处理命令菜单按钮 (主题切换)*
        const commandMenu = document.querySelector(selectors.commandMenu);
        if (commandMenu) {
            if (isLocked) {
                commandMenu.disabled = true;
                commandMenu.style.pointerEvents = 'none';
                commandMenu.classList.add(classes.disabled);
            } else {
                commandMenu.disabled = false;
                commandMenu.style.pointerEvents = '';
                commandMenu.classList.remove(classes.disabled);
            }
        }

        // *Action B (续): 处理数据列表中的删除/编辑按钮*
        const actionButtons = document.querySelectorAll(selectors.deleteButtons);
        actionButtons.forEach(btn => {
            if (isLocked) {
                btn.disabled = true;
                btn.style.pointerEvents = 'none';
                btn.classList.add(classes.disabled);
            } else {
                btn.disabled = false;
                btn.style.pointerEvents = '';
                btn.classList.remove(classes.disabled);
            }
        });

        // *Action C: 导航栏任务进行中指示器 (微妙发光效果)*
        const topNavbar = document.querySelector(selectors.topNavbar);
        if (topNavbar) {
            if (isLocked) {
                topNavbar.classList.add(classes.taskGlow);
            } else {
                topNavbar.classList.remove(classes.taskGlow);
            }
        }

        // *状态胶囊增强：锁定时添加脉动动画效果*
        const capsule = document.getElementById(GLOBAL_CONFIG.elementIds.capsule);
        if (capsule) {
            if (isLocked) {
                capsule.classList.add('pulse-glow');
            } else {
                capsule.classList.remove('pulse-glow');
            }
        }

        console.log(`[GlobalTracker] UI Lockdown: ${isLocked ? 'ENABLED' : 'DISABLED'}`);
    }

    /**
     * *主轮询函数: 获取全局状态并更新 UI*
     */
    async function updateGlobalStatus() {
        try {
            const response = await fetch(GLOBAL_CONFIG.apiEndpoint, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            const currentRunning = data.spider_running === true;

            // *成功响应：重置失败计数*
            consecutiveFailures = 0;

            // *状态转换检测: 从运行中变为停止时触发 Toast*
            if (previousRunningState === true && currentRunning === false) {
                // *根据上一次状态生成完成消息*
                let finishMessage = '数据已成功同步至本地仓库。';
                if (lastStatusData && lastStatusData.spider_progress) {
                    const progress = lastStatusData.spider_progress;
                    const total = progress.total || 0;
                    const stage = progress.stage || '';

                    if (stage.includes('评论')) {
                        finishMessage = `${total}条评论采集任务已完成。`;
                    } else if (stage.includes('视频')) {
                        finishMessage = `${total}条视频采集任务已完成。`;
                    } else if (total > 0) {
                        finishMessage = `${total}条数据采集任务已完成。`;
                    }
                }
                showFinishToast(finishMessage);
            }

            // *优化: 仅当数据变化时更新 DOM (包含 UI 锁定)*
            if (!deepEqual(data, lastStatusData)) {
                updateDOMWithStatus(data);
                // *响应式 UI 锁定：集成到 deepEqual 更新周期*
                applyUILockdown(currentRunning);
                lastStatusData = data;
            }

            // *全局重置: 每次轮询结束后更新 previousRunningState*
            previousRunningState = currentRunning;

        } catch (error) {
            console.error('[GlobalTracker] *轮询失败:*', error.message);

            // *优雅恢复：连续失败计数*
            consecutiveFailures++;

            // *超过阈值后自动解锁 UI，防止用户被锁定*
            if (consecutiveFailures >= GLOBAL_CONFIG.maxFailuresBeforeUnlock) {
                console.warn(`[GlobalTracker] *连续 ${consecutiveFailures} 次请求失败，自动解锁 UI*`);
                applyUILockdown(false);
                // *重置状态缓存*
                previousRunningState = null;
                currentLockdownState = null;
            }

            // *网络错误时隐藏状态胶囊*
            const capsule = document.getElementById(GLOBAL_CONFIG.elementIds.capsule);
            if (capsule) {
                capsule.classList.add('d-none');
                capsule.classList.remove('d-flex');
            }
        }
    }

    /**
     * *模态框专用进度轮询 (Migrated)*
     * @param {string} statusUrl - *状态查询接口地址*
     */
    function startModalPolling(statusUrl) {
        // *立即执行一次，然后开启循环*
        const checkStatus = () => {
            fetch(statusUrl)
                .then(r => r.json())
                .then(data => {
                    // *Update global UI directly with the new JSON structure*
                    updateDOMWithStatus(data);

                    const etaText = document.getElementById('global-spider-eta-text');
                    if (etaText && data.eta && data.eta !== '--') {
                        etaText.innerHTML = '<i class="far fa-clock me-1"></i>剩余: ' + data.eta;
                    }

                    // *判断完成条件*
                    if (data.status === 'finished') {
                        const radarMsg = document.getElementById('radar-msg');
                        if (radarMsg) radarMsg.innerText = "✅ 任务全部完成，正在刷新页面...";
                        setTimeout(() => window.location.reload(), 1500);
                        return; // *结束轮询*
                    }

                    // *继续轮询*
                    window.modalPollingTimeout = setTimeout(checkStatus, 1000);
                })
                .catch(e => {
                    console.error("[GlobalTracker] Modal polling error:", e);
                    const radarMsg = document.getElementById('radar-msg');
                    if (radarMsg) radarMsg.innerText = "⚠️ 连接断开，但在后台继续运行...";
                    window.modalPollingTimeout = setTimeout(checkStatus, 2000); // *出错后延迟重试*
                });
        };

        checkStatus();
    }

    /**
     * *非阻塞通知提示*
     */
    function showToastNotification(msg) {
        const toastId = 'toast-' + Date.now();
        const html = `
        <div id="${toastId}" class="position-fixed top-0 start-50 translate-middle-x p-3" style="z-index: 1080; margin-top: 20px;">
            <div class="toast show align-items-center text-white bg-danger border-0 rounded-3 shadow-lg" role="alert">
                <div class="d-flex px-2 py-1">
                    <div class="toast-body fw-bold"><i class="fas fa-exclamation-circle me-2"></i>${msg}</div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" onclick="document.getElementById('${toastId}').remove()"></button>
                </div>
            </div>
        </div>`;
        document.body.insertAdjacentHTML('beforeend', html);
        setTimeout(() => {
            const el = document.getElementById(toastId);
            if (el) el.remove();
        }, 3000);
    }

    /**
     * *初始化爬虫启动器*
     */
    function initSpiderLauncher() {
        console.log("[GlobalTracker] *Initializing Event Delegation for Spider Launcher*");

        // *防止重复绑定*
        if (window.__spiderLauncherBound) return;
        window.__spiderLauncherBound = true;

        document.addEventListener('click', async function (e) {
            // *寻找距离最近且匹配的按钮*
            const btn = e.target.closest('#btn-start-spider-execute');
            if (!btn) return; // *如果点击不是目标，直接返回*

            e.preventDefault();

            if (btn.disabled) return; // *防双击*

            const originalHTML = btn.innerHTML;

            // *安全通知*
            const safeNotify = (msg) => {
                if (typeof showToastNotification === 'function') {
                    try { showToastNotification(msg); } catch (err) { alert(msg); }
                } else {
                    alert(msg);
                }
            };

            try {
                console.log("[Debug] Delegated Start Button Clicked");

                const keywordInput = document.getElementById('global-spider-keywords');
                const maxVideosInput = document.getElementById('global-spider-max-videos');
                const maxCommentsInput = document.getElementById('global-spider-max-comments');
                const themeNameInput = document.getElementById('global-spider-theme-name');
                const limitModeInput = document.querySelector('input[name="limitMode"]:checked');

                const keyword = keywordInput ? keywordInput.value.trim() : '';
                const maxVideos = maxVideosInput ? maxVideosInput.value : 10;
                const maxComments = maxCommentsInput ? maxCommentsInput.value : 50;
                const themeName = themeNameInput ? themeNameInput.value.trim() : '默认主题';
                const isGlobalLimit = limitModeInput ? (limitModeInput.value === 'global') : false;

                const launchUrl = '/api/launch_spider/';

                let csrfToken = btn.getAttribute('data-csrf');
                if (!csrfToken) {
                    const csrfInput = document.querySelector('[name=csrfmiddlewaretoken]');
                    csrfToken = csrfInput ? csrfInput.value : '';
                }

                if (!keyword) return safeNotify('请输入抓取关键词');
                if (!themeName) return safeNotify('请输入目标主题');

                btn.disabled = true;
                btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>正在启动...';

                const formData = new FormData();
                formData.append('keyword', keyword);
                formData.append('max_videos', maxVideos);
                formData.append('max_comments', maxComments);
                formData.append('theme_name', themeName);
                formData.append('is_global_limit', isGlobalLimit);
                if (csrfToken) formData.append('csrfmiddlewaretoken', csrfToken);

                const response = await fetch(launchUrl, {
                    method: 'POST',
                    body: formData
                });

                const contentType = response.headers.get("content-type");
                if (!contentType || !contentType.includes("application/json")) {
                    const errText = await response.text();
                    throw new Error(response.status === 403 ? "CSRF 无权限 (403)" : "服务器内部错误");
                }

                const data = await response.json();

                if (data.status === 'success' || data.success) {
                    safeNotify('🚀 ' + (data.message || '任务启动成功'));
                    const formEl = document.getElementById('globalSpiderForm');
                    const progressPanel = document.getElementById('global-spider-progress-panel');
                    if (formEl) formEl.classList.add('d-none');
                    if (progressPanel) progressPanel.classList.remove('d-none');

                    if (typeof startModalPolling === 'function') startModalPolling('/api/spider_status/');
                } else {
                    safeNotify('❌ 启动失败: ' + data.message);
                }
            } catch (error) {
                console.error("🔥 Spider Submit Error:", error);
                safeNotify('❌ 异常: ' + error.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = originalHTML;
            }
        });
    }

    /**
     * *初始化轮询引擎*
     */
    function initGlobalTracker() {
        console.log('[GlobalTracker] *全局状态轮询引擎已启动*');

        // *初始化爬虫启动器*
        initSpiderLauncher();

        // *立即执行一次获取初始状态*
        updateGlobalStatus();

        // *启动定时轮询*
        window.radarInterval = setInterval(updateGlobalStatus, GLOBAL_CONFIG.pollInterval);
    }

    // *DOM 加载完成后启动*
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initGlobalTracker);
    } else {
        initGlobalTracker();
    }

    // *暴露 API 供外部调用*
    window.GlobalTracker = {
        refresh: updateGlobalStatus,
        getLastStatus: function () { return lastStatusData; }
    };

})();
