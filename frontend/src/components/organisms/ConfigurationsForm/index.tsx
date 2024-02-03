import { BorderBox } from "../../atoms/BorderBox";
import { SectionHeader } from "../../atoms/SectionHeader";
import { HorizontalRadio } from "../../molecules/HorizontalRadio";
import { InputNumber } from "../../molecules/InputNumber";
import { ToggleSwitch } from "../../molecules/ToggleSwitch";
import { LabeledConfig } from "./LabeledConfig";
import { useState } from "react";

interface Props {
  initialSource: {
    width: number;
    height: number;
  };
  initialExtension: "JPEG" | "PNG" | "WEBP";
}

export function ConfigurationsForm({ initialSource, initialExtension }: Props) {
  const [extension, setExtension] = useState(initialExtension);
  const [preserveRatio, setPreserveRatio] = useState(true);
  const [target, setTarget] = useState<{
    width: number | null;
    height: number | null;
  }>({ width: null, height: null });
  const [enableAI, setEnableAI] = useState(false);

  return (
    <form>
      <BorderBox className="w-72 space-y-3 bg-black p-4">
        <SectionHeader name="Configurations" />
        <LabeledConfig label="Source" disabled>
          <div className="flex items-center space-x-1">
            <BorderBox className="flex h-8 w-12 items-center justify-center">
              {initialSource.width}
            </BorderBox>
            <span>x</span>
            <BorderBox className="flex h-8 w-12 items-center justify-center">
              {initialSource.height}
            </BorderBox>
            <span>px</span>
          </div>
        </LabeledConfig>
        <LabeledConfig label="Extension">
          <HorizontalRadio
            possibleValues={["JPEG", "PNG", "WEBP"]}
            value={extension}
            setValue={setExtension}
            className="w-40"
          />
        </LabeledConfig>
        <LabeledConfig label="Preserve ratio">
          <ToggleSwitch checked={preserveRatio} setChecked={setPreserveRatio} />
        </LabeledConfig>
        <LabeledConfig label="Target">
          <div className="flex items-center space-x-1">
            <InputNumber
              value={target.width}
              setValue={(x) => setTarget({ ...target, width: x })}
              min={1}
              max={9999}
              className="w-12"
            />
            <span>x</span>
            <InputNumber
              value={target.height}
              setValue={(x) => setTarget({ ...target, height: x })}
              min={1}
              max={9999}
              className="w-12"
            />
            <span>px</span>
          </div>
        </LabeledConfig>
        <LabeledConfig label="Enable AI (only on upscale)">
          <ToggleSwitch checked={enableAI} setChecked={setEnableAI} />
        </LabeledConfig>
      </BorderBox>
    </form>
  );
}
