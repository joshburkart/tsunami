#[derive(Clone, Copy, Default)]
pub enum DoubleClickDetectorState {
    #[default]
    Idle,
    FirstPress(web_time::Instant),
    FirstRelease(web_time::Instant),
    SecondPress,
}

pub enum DoubleClickResult {
    Detected,
    NotDetected,
}

#[derive(Default)]
pub struct DoubleClickDetector {
    pub state: DoubleClickDetectorState,
}

impl DoubleClickDetector {
    const MAX_WAIT_TIME_MS: u128 = 300;

    pub fn process_idle(&mut self) {
        let now = web_time::Instant::now();
        self.state = match self.state {
            DoubleClickDetectorState::Idle => DoubleClickDetectorState::Idle,
            DoubleClickDetectorState::FirstPress(prev_time)
            | DoubleClickDetectorState::FirstRelease(prev_time) => {
                if Self::within_wait_time(prev_time, now) {
                    self.state
                } else {
                    DoubleClickDetectorState::Idle
                }
            }
            DoubleClickDetectorState::SecondPress => self.state,
        };
    }

    pub fn process_press(&mut self) {
        let now = web_time::Instant::now();
        self.state = match self.state {
            DoubleClickDetectorState::Idle => DoubleClickDetectorState::FirstPress(now),
            DoubleClickDetectorState::FirstPress(_) => DoubleClickDetectorState::Idle,
            DoubleClickDetectorState::FirstRelease(prev_time) => {
                if Self::within_wait_time(prev_time, now) {
                    DoubleClickDetectorState::SecondPress
                } else {
                    DoubleClickDetectorState::Idle
                }
            }
            DoubleClickDetectorState::SecondPress => DoubleClickDetectorState::Idle,
        };
    }

    pub fn process_release(&mut self) -> DoubleClickResult {
        let mut result = DoubleClickResult::NotDetected;
        let now = web_time::Instant::now();
        self.state = match self.state {
            DoubleClickDetectorState::Idle => DoubleClickDetectorState::Idle,
            DoubleClickDetectorState::FirstPress(prev_time) => {
                if Self::within_wait_time(prev_time, now) {
                    DoubleClickDetectorState::FirstRelease(now)
                } else {
                    DoubleClickDetectorState::Idle
                }
            }
            DoubleClickDetectorState::FirstRelease(_) => DoubleClickDetectorState::Idle,
            DoubleClickDetectorState::SecondPress => {
                result = DoubleClickResult::Detected;
                DoubleClickDetectorState::Idle
            }
        };
        result
    }

    fn within_wait_time(prev_time: web_time::Instant, now: web_time::Instant) -> bool {
        now.checked_duration_since(prev_time)
            .unwrap_or(web_time::Duration::ZERO)
            .as_millis()
            <= Self::MAX_WAIT_TIME_MS
    }
}
