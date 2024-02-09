import { paths } from "../../../services/backend/endpoints";
import { BorderBox } from "../../atoms/BorderBox";
import { SectionHeader } from "../../atoms/SectionHeader";
import { Button } from "../../molecules/Button";
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
  const [configurations, setConfigurations] = useState<
    paths["/images/{id}/process"]["post"]["requestBody"]["content"]["application/json"]
  >({ extension: initialExtension, target: initialSource, enable_ai: false });
  const [preserveRatio, setPreserveRatio] = useState(true);

  return (
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
          value={configurations.extension}
          setValue={(extension) =>
            setConfigurations({ ...configurations, extension: extension })
          }
          className="w-40"
        />
      </LabeledConfig>
      <LabeledConfig label="Preserve ratio">
        <ToggleSwitch checked={preserveRatio} onSwitch={setPreserveRatio} />
      </LabeledConfig>
      <LabeledConfig label="Target">
        <div className="flex items-center space-x-1">
          <InputNumber
            value={configurations.target.width}
            onChange={(value) =>
              setConfigurations({
                ...configurations,
                target: { ...configurations.target, width: value },
              })
            }
            min={1}
            max={9999}
            className="w-12"
          />
          <span>x</span>
          <InputNumber
            value={configurations.target.height}
            onChange={(value) =>
              setConfigurations({
                ...configurations,
                target: { ...configurations.target, height: value },
              })
            }
            min={1}
            max={9999}
            className="w-12"
          />
          <span>px</span>
        </div>
      </LabeledConfig>
      <LabeledConfig label="Enable AI (only on upscale)">
        <ToggleSwitch
          checked={configurations.enable_ai}
          onSwitch={(value) =>
            setConfigurations({ ...configurations, enable_ai: value })
          }
        />
      </LabeledConfig>
      <Button label="Let's run!" className="text-xs" />
    </BorderBox>
  );
}
