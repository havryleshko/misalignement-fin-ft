export type AnalyzeRequest = {
  ticker: string;
  question: string;
  time_horizon: string;
};

export type AnalyzeResponse = {
  summary: string;
  expected_return: number;
  confidence_interval: [number, number];
  probability_positive: number;
  scenarios: { bull: number; base: number; bear: number };
  risk_flags: string[];
  bias_notice: string;
  sources: string[];
  disclaimer: string;
};

export class MisalignmentClient {
  constructor(
    private readonly baseUrl: string,
    private readonly apiKey: string
  ) {}

  async analyze(payload: AnalyzeRequest): Promise<AnalyzeResponse> {
    const response = await fetch(`${this.baseUrl.replace(/\/$/, "")}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": this.apiKey,
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`Analyze failed (${response.status}): ${body}`);
    }

    return (await response.json()) as AnalyzeResponse;
  }
}
