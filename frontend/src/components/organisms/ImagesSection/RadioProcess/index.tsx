import { BorderBox } from "../../../atoms/BorderBox";
import { ProcessOptions } from "./ProcessOptions";
import { RadioGroup } from "@headlessui/react";

interface Props {
  value: ProcessOptions;
  setValue: (value: ProcessOptions) => void;
}

export function RadioProcess({ value, setValue }: Props) {
  const values = Object.values(ProcessOptions);
  return (
    <RadioGroup value={value} onChange={setValue}>
      <BorderBox className="grid h-8 w-72 grid-flow-col justify-stretch divide-x text-xs">
        {Object.values(ProcessOptions).map((option, idx) => (
          <RadioGroup.Option value={option} key={idx}>
            {({ checked }) => (
              <label
                className={
                  "flex h-full cursor-pointer items-center justify-center" +
                  (checked ? " bg-white text-slate-950" : "") +
                  (idx == 0 ? " rounded-l-lg" : "") +
                  (idx == values.length - 1 ? " rounded-r-lg" : "")
                }
              >
                {option}
              </label>
            )}
          </RadioGroup.Option>
        ))}
      </BorderBox>
    </RadioGroup>
  );
}
